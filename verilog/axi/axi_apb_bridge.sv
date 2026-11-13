// AXI4-Lite → APB 桥（单事务串行化，单状态机实现）
//
// 通过 APB 连接慢速寄存器堆外设（apb_slave_regs，可配 wait states）。
// 支持：读 / 写 / 地址越界 SLVERR → BRESP/RRESP=2'b10。
//
// 合法性的关键点：
//   - AXI 五通道握手全部按 VALID/READY 法则（READY 只在对应状态拉高；
//     同拍 AW+AR 冲突用优先级避免双accept）
//   - APB 时序：SETUP（PSEL=1,PENABLE=0）至少 1 拍 → ACCESS（PSEL=1,
//     PENABLE=1）等 PREADY（慢速从机插 wait states，即"可以慢"）→ 完成
//   - 响应（BRESP/RVALID/RDATA/RRESP）在完成拍后给出，绝不提前
//
// 状态机：IDLE →（写）AW_HELD/W_HELD → W_SETUP → W_ACCESS → W_RESP → IDLE
//              →（读）R_SETUP → R_ACCESS → R_RESP → IDLE
`timescale 1ns / 1ps

module axi_apb_bridge #(
    parameter AW   = 8,    // 地址位宽
    parameter DW   = 32,   // 数据位宽
    parameter NREG = 16    // 外设寄存器数（地址 0..NREG*4-4）
)(
    input  wire          ACLK,
    input  wire          ARESETn,

    // AXI4-Lite 写地址通道
    output reg           AWREADY,
    input  wire          AWVALID,
    input  wire [AW-1:0] AWADDR,

    // AXI4-Lite 写数据通道
    output reg           WREADY,
    input  wire          WVALID,
    input  wire [DW-1:0] WDATA,

    // AXI4-Lite 写响应通道
    output reg           BVALID,
    input  wire          BREADY,
    output reg  [1:0]    BRESP,

    // AXI4-Lite 读地址通道
    output reg           ARREADY,
    input  wire          ARVALID,
    input  wire [AW-1:0] ARADDR,

    // AXI4-Lite 读数据通道
    output reg           RVALID,
    input  wire          RREADY,
    output reg  [DW-1:0] RDATA,
    output reg  [1:0]    RRESP,

    // APB 主侧（连慢速外设）
    output reg           PSEL,
    output reg           PENABLE,
    output reg           PWRITE,
    output reg  [AW-1:0] PADDR,
    output reg  [DW-1:0] PWDATA,
    input  wire          PREADY,
    input  wire [DW-1:0] PRDATA,
    input  wire          PSLVERR
);

    // ---- 状态编码 ----
    localparam IDLE     = 4'd0;
    localparam AW_HELD  = 4'd1;  // 收到地址，等写数据
    localparam W_HELD   = 4'd2;  // 收到数据，等写地址
    localparam W_SETUP  = 4'd3;  // APB SETUP（写）
    localparam W_ACCESS = 4'd4;  // APB ACCESS（写），等 PREADY
    localparam W_RESP   = 4'd5;  // 写响应 B 通道
    localparam R_SETUP  = 4'd6;  // APB SETUP（读）
    localparam R_ACCESS = 4'd7;  // APB ACCESS（读），等 PREADY
    localparam R_RESP   = 4'd8;  // 读响应 R 通道

    reg [3:0] state;

    // ---- 锁存的地址 / 数据 / 错误 ----
    reg [AW-1:0] paddr_q;
    reg [DW-1:0] pwdata_q;
    reg [DW-1:0] rdata_q;
    reg          slv_err;   // 最近一次 APB 事务的 PSLVERR

    // ==== 组合：AXI 握手 + APB 输出 ====
    always @* begin
        // 默认
        AWREADY = 1'b0; WREADY = 1'b0; ARREADY = 1'b0;
        BVALID  = 1'b0; RVALID  = 1'b0;
        BRESP   = 2'b00; RRESP   = 2'b00;
        PSEL    = 1'b0;  PENABLE = 1'b0; PWRITE  = 1'b0;
        PADDR   = paddr_q;  // 默认输出锁存地址（仅 APB 阶段有意义）
        PWDATA  = pwdata_q; // 默认输出锁存数据

        case (state)
            IDLE: begin
                // 优先写；同拍 AW+W 一起接受；读单发
                if (AWVALID && WVALID) begin
                    AWREADY = 1'b1; WREADY = 1'b1;
                end else if (AWVALID) begin
                    AWREADY = 1'b1;
                end else if (WVALID) begin
                    WREADY = 1'b1;
                end else if (ARVALID) begin
                    ARREADY = 1'b1;
                end
            end
            AW_HELD: WREADY = 1'b1;
            W_HELD:  AWREADY = 1'b1;
            W_SETUP:  begin PSEL = 1'b1; PENABLE = 1'b0; PWRITE = 1'b1; end
            W_ACCESS: begin PSEL = 1'b1; PENABLE = 1'b1; PWRITE = 1'b1; end
            W_RESP:   begin BVALID = 1'b1; BRESP = slv_err ? 2'b10 : 2'b00; end
            R_SETUP:  begin PSEL = 1'b1; PENABLE = 1'b0; end
            R_ACCESS: begin PSEL = 1'b1; PENABLE = 1'b1; end
            R_RESP:   begin RVALID = 1'b1;
                             RRESP = slv_err ? 2'b10 : 2'b00;
                             RDATA = rdata_q;
                       end
            default: ;
        endcase
    end

    // ==== 时序：状态转移 + 锁存 ====
    always @(posedge ACLK or negedge ARESETn) begin
        if (!ARESETn) begin
            state    <= IDLE;
            paddr_q  <= {AW{1'b0}};
            pwdata_q <= {DW{1'b0}};
            rdata_q  <= {DW{1'b0}};
            slv_err  <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    if (AWVALID && WVALID) begin
                        paddr_q <= AWADDR; pwdata_q <= WDATA;
                        state   <= W_SETUP;
                    end else if (AWVALID) begin
                        paddr_q <= AWADDR; state <= AW_HELD;
                    end else if (WVALID) begin
                        pwdata_q <= WDATA; state <= W_HELD;
                    end else if (ARVALID) begin
                        paddr_q <= ARADDR; state <= R_SETUP;
                    end
                end
                AW_HELD: begin
                    if (WVALID) begin
                        pwdata_q <= WDATA; state <= W_SETUP;
                    end
                end
                W_HELD: begin
                    if (AWVALID) begin
                        paddr_q <= AWADDR; state <= W_SETUP;
                    end
                end
                W_SETUP:  state <= W_ACCESS;
                W_ACCESS: begin
                    if (PREADY) begin
                        if (PSLVERR) slv_err <= 1'b1;
                        state <= W_RESP;
                    end
                end
                W_RESP: begin
                    if (BREADY) begin
                        slv_err <= 1'b0;   // 清除错误标记
                        state   <= IDLE;
                    end
                end
                R_SETUP:  state <= R_ACCESS;
                R_ACCESS: begin
                    if (PREADY) begin
                        if (PSLVERR) slv_err <= 1'b1;
                        rdata_q <= PRDATA;   // 数据在完成拍锁存
                        state   <= R_RESP;
                    end
                end
                R_RESP: begin
                    if (RREADY) begin
                        slv_err <= 1'b0;
                        state   <= IDLE;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end

endmodule