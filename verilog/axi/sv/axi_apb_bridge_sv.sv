// AXI4-Lite → APB 桥（完整 SV 接口端口版）
// 状态机串行化：同一时刻一个事务
//   - AXI master 侧：modport master（slave 角色）
//   - APB slave 侧：modport slave（master 角色）
//   单状态机：IDLE → (AW/W 握手) → APB WR → B → IDLE
//                    → (AR 握手)   → APB RD → R → IDLE
// "可以慢"：APB ACCESS 阶段等 PREADY（wait states 由从机控制）
// "合法"：AXI 握手法则、APB SETUP/ACCESS 时序、响应完成拍后给出
`ifndef AXISV_APB_BRIDGE_SV
`define AXISV_APB_BRIDGE_SV
`timescale 1ns / 1ps

module axi_apb_bridge_sv #(
    parameter AW = 8,
    parameter DW = 32
)(
    input  logic ACLK,
    input  logic ARESETn,
    axi_lite_if.slave  axi,   // 连 AXI master
    apb_if.master      apb    // 连 APB slave
);

    // ---- 状态 ----
    typedef enum logic [3:0] {
        S_IDLE,
        S_AW_H, S_W_H, S_AWW_H,
        S_WR_SETUP, S_WR_ACC, S_B_RESP,
        S_AR_H,
        S_RD_SETUP, S_RD_ACC, S_R_RESP
    } state_t;
    state_t state, next;

    // ---- 状态转移记录（供 TB 做转移覆盖统计）----
    // 覆盖「状态被访问」不等于覆盖「转移发生过」：
    // 例如 S_AW_H 与 S_W_H 都会被访问，但 S_W_H→S_AWW_H 这条
    // 只有「W 先到、AW 后到」时才会走。用 14 位掩码记录各转移是否发生。
    // 位序：IDLE→AWW/IDLE→AW/IDLE→W/IDLE→AR/AW_H→AWW/W_H→AWW/
    //       AWW→WR_SETUP/WR_SETUP→WR_ACC/WR_ACC→B_RESP/B_RESP→IDLE/
    //       AR_H→RD_SETUP/RD_SETUP→RD_ACC/RD_ACC→R_RESP/R_RESP→IDLE
    logic [13:0] trans_seen;
    always_ff @(posedge ACLK or negedge ARESETn) begin
        if (!ARESETn) trans_seen <= '0;
        else begin
            case ({state, next})
                {S_IDLE,     S_AWW_H}   : trans_seen[0]  <= 1'b1;
                {S_IDLE,     S_AW_H}    : trans_seen[1]  <= 1'b1;
                {S_IDLE,     S_W_H}     : trans_seen[2]  <= 1'b1;
                {S_IDLE,     S_AR_H}    : trans_seen[3]  <= 1'b1;
                {S_AW_H,     S_AWW_H}   : trans_seen[4]  <= 1'b1;
                {S_W_H,      S_AWW_H}   : trans_seen[5]  <= 1'b1;
                {S_AWW_H,    S_WR_SETUP}: trans_seen[6]  <= 1'b1;
                {S_WR_SETUP, S_WR_ACC}  : trans_seen[7]  <= 1'b1;
                {S_WR_ACC,   S_B_RESP}  : trans_seen[8]  <= 1'b1;
                {S_B_RESP,   S_IDLE}    : trans_seen[9]  <= 1'b1;
                {S_AR_H,     S_RD_SETUP}: trans_seen[10] <= 1'b1;
                {S_RD_SETUP, S_RD_ACC}  : trans_seen[11] <= 1'b1;
                {S_RD_ACC,   S_R_RESP}  : trans_seen[12] <= 1'b1;
                {S_R_RESP,   S_IDLE}    : trans_seen[13] <= 1'b1;
                default: ;
            endcase
        end
    end

    // ---- 锁存 ----
    logic [AW-1:0] paddr_q;
    logic [DW-1:0] pwdata_q;
    logic [DW-1:0] rdata_q;
    logic          slv_err;

    // ---- AXI 握手输出（组合） ----
    always_comb begin
        axi.AWREADY = 1'b0;
        axi.WREADY  = 1'b0;
        axi.ARREADY = 1'b0;
        axi.BVALID  = 1'b0;
        axi.BRESP   = axi_defs_pkg::RESP_OKAY;
        axi.RVALID  = 1'b0;
        axi.RRESP   = axi_defs_pkg::RESP_OKAY;
        axi.RDATA   = rdata_q;

        case (state)
            S_IDLE: begin
                // 同拍 AW+W 一起接受；或先 AW 后 W / 先 W 后 AW / 仅 AR
                if (axi.AWVALID && axi.WVALID) begin
                    axi.AWREADY = 1'b1; axi.WREADY = 1'b1;
                end else if (axi.AWVALID) begin
                    axi.AWREADY = 1'b1;
                end else if (axi.WVALID) begin
                    axi.WREADY = 1'b1;
                end else if (axi.ARVALID) begin
                    axi.ARREADY = 1'b1;
                end
            end
            // 握手阶段保持双方 READY 拉高（AXI 允许），消除"单拍可见"竞态：
            // 已接受的一侧继续拉高，避免主站晚一拍撤销 VALID 时被判为中途撤销。
            S_AW_H: begin axi.WREADY = 1'b1; axi.AWREADY = 1'b1; end  // 已收 AW，等 W
            S_W_H:  begin axi.AWREADY = 1'b1; axi.WREADY = 1'b1; end  // 已收 W，等 AW
            S_AWW_H: begin
                // 已收齐 AW+W，本拍仍保持 READY（收集完成），下一拍进 APB
                axi.AWREADY = 1'b1; axi.WREADY = 1'b1;
            end
            S_B_RESP: begin
                axi.BVALID = 1'b1;
                axi.BRESP  = slv_err ? axi_defs_pkg::RESP_SLVERR : axi_defs_pkg::RESP_OKAY;
            end
            S_R_RESP: begin
                axi.RVALID = 1'b1;
                axi.RRESP  = slv_err ? axi_defs_pkg::RESP_SLVERR : axi_defs_pkg::RESP_OKAY;
            end
            default: ;
        endcase
    end

    // ---- APB 输出（组合） ----
    always_comb begin
        apb.PSEL    = 1'b0;
        apb.PENABLE = 1'b0;
        apb.PWRITE  = 1'b0;
        apb.PADDR   = paddr_q;
        apb.PWDATA  = pwdata_q;

        case (state)
            S_WR_SETUP: begin apb.PSEL = 1'b1; apb.PENABLE = 1'b0; apb.PWRITE = 1'b1; end
            S_WR_ACC:   begin apb.PSEL = 1'b1; apb.PENABLE = 1'b1; apb.PWRITE = 1'b1; end
            S_RD_SETUP: begin apb.PSEL = 1'b1; apb.PENABLE = 1'b0; end
            S_RD_ACC:   begin apb.PSEL = 1'b1; apb.PENABLE = 1'b1; end
            default: ;
        endcase
    end

    // ---- 状态转移（组合） ----
    always_comb begin
        next = state;
        case (state)
            S_IDLE: begin
                if (axi.AWVALID && axi.WVALID)      next = S_AWW_H;
                else if (axi.AWVALID)               next = S_AW_H;
                else if (axi.WVALID)                next = S_W_H;
                else if (axi.ARVALID)               next = S_AR_H;
            end
            S_AW_H:   if (axi.WVALID)               next = S_AWW_H;
            S_W_H:    if (axi.AWVALID)              next = S_AWW_H;
            S_AWW_H:  next = S_WR_SETUP;
            S_WR_SETUP: next = S_WR_ACC;
            S_WR_ACC: if (apb.PREADY)               next = S_B_RESP;
            S_B_RESP: if (axi.BREADY)               next = S_IDLE;
            S_AR_H:   next = S_RD_SETUP;
            S_RD_SETUP: next = S_RD_ACC;
            S_RD_ACC: if (apb.PREADY)               next = S_R_RESP;
            S_R_RESP: if (axi.RREADY)               next = S_IDLE;
            default: next = S_IDLE;
        endcase
    end

    // ---- 错误标记（单一时序过程） ----
    // 置位：APB 完成拍抓到 PSLVERR；清除：响应握手拍
    always_ff @(posedge ACLK or negedge ARESETn) begin
        if (!ARESETn)
            slv_err <= 1'b0;
        else begin
            // 完成拍置位（读写共用）
            if ((state == S_WR_ACC && apb.PREADY && apb.PSLVERR) ||
                (state == S_RD_ACC && apb.PREADY && apb.PSLVERR))
                slv_err <= 1'b1;
            else if (state == S_B_RESP && axi.BREADY)
                slv_err <= 1'b0;
            else if (state == S_R_RESP && axi.RREADY)
                slv_err <= 1'b0;
        end
    end

    // ---- 读数据锁存（单一时序过程） ----
    always_ff @(posedge ACLK or negedge ARESETn) begin
        if (!ARESETn)
            rdata_q <= '0;
        else if (state == S_RD_ACC && apb.PREADY)
            rdata_q <= apb.PRDATA;
    end

    // ---- 时序：锁存 + 状态 ----
    always_ff @(posedge ACLK or negedge ARESETn) begin
        if (!ARESETn) begin
            state    <= S_IDLE;
            paddr_q  <= '0;
            pwdata_q <= '0;
            // rdata_q / slv_err 由各自独立时序过程管理
        end else begin
            // 锁存地址/数据
            case (state)
                S_IDLE: begin
                    if (axi.AWVALID && axi.WVALID) begin
                        paddr_q <= axi.AWADDR;
                        pwdata_q <= axi.WDATA;
                    end else if (axi.AWVALID) begin
                        paddr_q <= axi.AWADDR;
                    end else if (axi.WVALID) begin
                        pwdata_q <= axi.WDATA;
                    end else if (axi.ARVALID) begin
                        paddr_q <= axi.ARADDR;
                    end
                end
                S_AW_H: if (axi.WVALID) pwdata_q <= axi.WDATA;
                S_W_H:  if (axi.AWVALID) paddr_q <= axi.AWADDR;
                default: ;
            endcase
            state <= next;
        end
    end

endmodule
`endif // AXISV_APB_BRIDGE_SV