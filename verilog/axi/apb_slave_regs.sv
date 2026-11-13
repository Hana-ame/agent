// APB slave：慢速寄存器堆外设（带 wait states + 地址越界 SLVERR）
//
// 协议要点（APB v2.0，完整定义见 apb_defs_pkg.sv）：
//   - 事务分 SETUP（PSEL=1, PENABLE=0）和 ACCESS（PSEL=1, PENABLE=1）两阶段
//   - ACCESS 阶段 PREADY=0 时插入 wait state（"可以慢"）
//   - PREADY=1 且 PSEL=1 且 PENABLE=1 时事务完成（apb_defs_pkg::transfer_done）
//     （读：PRDATA 有效；写：采样写入；PSLVERR 可选置位表示错误）
//   - 地址越界（>= NREG*4）返回 PSLVERR=1，不写存储器
//   - 本模块用 apb_defs_pkg::apb_phase_t 枚举实现阶段状态机（IDLE/SETUP/ACC）
`timescale 1ns / 1ps

module apb_slave_regs #(
    parameter AW          = 8,   // 地址位宽
    parameter DW          = 32,  // 数据位宽
    parameter NREG        = 16,  // 寄存器个数（地址 0..NREG*4-4）
    parameter WAIT_CYCLES = 2    // ACCESS 阶段插入的 wait states 数
)(
    input  wire             PCLK,
    input  wire             PRESETn,
    input  wire             PSEL,
    input  wire             PENABLE,
    input  wire             PWRITE,
    input  wire [AW-1:0]    PADDR,
    input  wire [DW-1:0]    PWDATA,
    output reg              PREADY,
    output reg  [DW-1:0]    PRDATA,
    output reg              PSLVERR
);

    // ---- 寄存器堆（写使能由 PWRITE 在完成拍控制） ----
    reg [DW-1:0] mem [0:NREG-1];

    integer i;
    // 上电清零（保证仿真从确定状态开始）
    initial begin
        for (i = 0; i < NREG; i = i + 1) mem[i] = {DW{1'b0}};
    end

    // ---- 地址译码 ----
    wire [AW-1:0] idx   = PADDR >> 2;              // 字序号
    wire          in_range = (idx < NREG);

    // ---- 导入 APB 定义层 ----
    import apb_defs_pkg::*;

    // ---- 阶段状态机（用 apb_defs_pkg::apb_phase_t 显式枚举） ----
    apb_phase_t phase;     // IDLE / SETUP / ACC
    reg [7:0]   wait_cnt;  // ACCESS 阶段剩余 wait states

    always @(posedge PCLK or negedge PRESETn) begin
        if (!PRESETn) begin
            phase    <= APB_IDLE;
            wait_cnt <= 8'h0;
            PREADY   <= 1'b0;
            PSLVERR  <= 1'b0;
            PRDATA   <= {DW{1'b0}};
        end else begin
            case (phase)
                APB_IDLE: begin
                    PREADY  <= 1'b0;
                    PSLVERR <= 1'b0;
                    // 进入 SETUP：登记 wait 计数（R: SETUP 至少 1 拍）
                    if (PSEL && !PENABLE) begin
                        wait_cnt <= WAIT_CYCLES[7:0];
                        phase    <= APB_SETUP;
                    end
                end
                APB_SETUP: begin
                    PREADY  <= 1'b0;   // R3: SETUP 期间从机不回 PREADY
                    PSLVERR <= 1'b0;
                    // 桥拉高 PENABLE -> 进入 ACCESS
                    if (PENABLE)
                        phase <= APB_ACC;
                end
                APB_ACC: begin
                    if (wait_cnt == 0) begin
                        // wait 耗尽：本拍完成（transfer_done 判据）
                        PREADY  <= 1'b1;
                        PSLVERR <= in_range ? 1'b0 : 1'b1;   // 越界报错
                        if (in_range && PWRITE)
                            mem[idx] <= PWDATA;              // 写：完成拍锁存
                        if (!PWRITE)
                            PRDATA <= in_range ? mem[idx] : {DW{1'b0}};
                        phase <= APB_IDLE;                   // 完成本传输
                    end else begin
                        // 插入 wait state（"可以慢"）：PREADY=0 保持 ACCESS
                        PREADY   <= 1'b0;
                        wait_cnt <= wait_cnt - 1'b1;
                    end
                end
                default: begin
                    phase    <= APB_IDLE;
                    PREADY   <= 1'b0;
                    PSLVERR  <= 1'b0;
                end
            endcase
        end
    end

endmodule