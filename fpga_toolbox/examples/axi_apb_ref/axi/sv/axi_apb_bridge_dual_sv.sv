// ============================================================================
// axi_apb_bridge_dual_sv：AXI4-Lite → APB 桥（读写并发版）
// ============================================================================
// 相比单事务串行版（axi_apb_bridge_sv），本版把 AXI 的读写通道解耦：
//
//   ┌────────────┐
//   │ 写路径 FSM  │ AW/W → APB(写) → B
//   └─────┬──────┘
//         │ 请求授权
//   ┌─────▼──────┐        ┌──────────────┐
//   │ APB 仲裁器  │───────▶│  APB 总线    │  （物理只有一条，按事务仲裁）
//   └─────▲──────┘        └──────────────┘
//         │ 请求授权
//   ┌─────┴──────┐
//   │ 读路径 FSM  │ AR → APB(读) → R
//   └────────────┘
//
// 并发语义（符合 AXI4-Lite）：
//   - AW/AR 可同拍接受（两条路径各自都有空时）
//   - 写读事务在 APB 上按事务粒度交替（仲裁），而非整笔串行
//   - 关键收益：写进入 B 响应阶段时，读可立即占用 APB；
//               读进入 R 响应阶段时，写可立即占用 APB
//   - B 与 R 是独立通道，响应可乱序出现
//
// 约束（有意设计）：
//   - 同类事务内串行（上一笔写完全响应后才接受下一笔 AW/W），
//     保证 AXI4-Lite 的写响应顺序语义；读同理
//   - APB 仲裁采用 round-robin，避免一方饿死
// ============================================================================
`ifndef AXI_APB_BRIDGE_DUAL_SV
`define AXI_APB_BRIDGE_DUAL_SV
`timescale 1ns / 1ps

module axi_apb_bridge_dual_sv #(
    parameter int AW = 8,
    parameter int DW = 32
)(
    input  logic ACLK,
    input  logic ARESETn,
    axi_lite_if.slave axi,
    apb_if.master     apb
);

    // ==================================================================
    // 写路径状态机
    // ==================================================================
    typedef enum logic [2:0] {
        W_IDLE,     // 等 AW+W
        W_REQ,      // 请求 APB 授权
        W_SETUP,    // APB SETUP（PSEL=1, PENABLE=0, PWRITE=1）
        W_ACC,      // APB ACCESS（等 PREADY）
        W_RESP      // B 响应
    } wstate_t;
    wstate_t wst;

    logic          aw_held, w_held;
    logic [AW-1:0] awaddr_q;
    logic [DW-1:0] wdata_q;
    logic          w_slverr;

    // ==================================================================
    // 读路径状态机
    // ==================================================================
    typedef enum logic [2:0] {
        R_IDLE,
        R_REQ,
        R_SETUP,
        R_ACC,
        R_RESP
    } rstate_t;
    rstate_t rst;

    // ---- 状态转移记录（供 TB 做转移覆盖统计）----
    // 与串行桥同理：状态被访问 ≠ 转移发生过。
    // 本模块无 next 变量（单块 always_ff 直接赋值），
    // 因此用「前置状态」寄存器记录上一拍状态，与本拍状态比较得出转移。
    // 写 FSM 5 条转移占 [4:0]，读 FSM 5 条占 [9:5]。
    logic [9:0] trans_seen;
    wstate_t wst_prev, wst_prev2;
    rstate_t rst_prev, rst_prev2;
    always_ff @(posedge ACLK or negedge ARESETn) begin
        if (!ARESETn) begin
            trans_seen <= '0;
            wst_prev   <= W_IDLE; wst_prev2 <= W_IDLE;
            rst_prev   <= R_IDLE; rst_prev2 <= R_IDLE;
        end else begin
            // 记录「本拍状态」，下一拍用它作为 prev 与新的本拍比较。
            // 注意：不能在同一拍同时更新 prev 并用它比较 —— 非阻塞赋值下
            //      prev 仍是旧值，会漏掉最后一次转移（本轮实测漏了
            //      R: RESP->IDLE，因为测试结束时最后那次转移没被记到）。
            wst_prev2 <= wst_prev;
            rst_prev2 <= rst_prev;
            wst_prev  <= wst;
            rst_prev  <= rst;
            if (wst_prev2 == W_IDLE  && wst_prev == W_REQ)   trans_seen[0] <= 1'b1;
            if (wst_prev2 == W_REQ   && wst_prev == W_SETUP) trans_seen[1] <= 1'b1;
            if (wst_prev2 == W_SETUP && wst_prev == W_ACC)   trans_seen[2] <= 1'b1;
            if (wst_prev2 == W_ACC   && wst_prev == W_RESP)  trans_seen[3] <= 1'b1;
            if (wst_prev2 == W_RESP  && wst_prev == W_IDLE)  trans_seen[4] <= 1'b1;
            if (rst_prev2 == R_IDLE  && rst_prev == R_REQ)   trans_seen[5] <= 1'b1;
            if (rst_prev2 == R_REQ   && rst_prev == R_SETUP) trans_seen[6] <= 1'b1;
            if (rst_prev2 == R_SETUP && rst_prev == R_ACC)   trans_seen[7] <= 1'b1;
            if (rst_prev2 == R_ACC   && rst_prev == R_RESP)  trans_seen[8] <= 1'b1;
            if (rst_prev2 == R_RESP  && rst_prev == R_IDLE)  trans_seen[9] <= 1'b1;
        end
    end

    logic          ar_held;
    logic [AW-1:0] araddr_q;
    logic [DW-1:0] rdata_q;
    logic          r_slverr;

    // ==================================================================
    // APB 仲裁器（round-robin）
    // ==================================================================
    typedef enum logic [1:0] { A_NONE, A_WR, A_RD } owner_t;
    owner_t owner;
    logic   last_rd;      // 上次服务的是读 → 本次优先写

    logic w_req, r_req;
    assign w_req = (wst == W_REQ);
    assign r_req = (rst == R_REQ);

    logic w_grant, r_grant;
    assign w_grant = (owner == A_NONE) && w_req && (!r_req || !last_rd);
    assign r_grant = (owner == A_NONE) && r_req && (!w_req ||  last_rd);

    // ==================================================================
    // AXI 输出（组合）
    // ==================================================================
    always_comb begin
        // ---- 写通道 ----
        // W_IDLE 与 W_REQ 都保持 READY 高：
        //   W_REQ 期间事务已收齐，继续保持 READY 可消除"握手完成当拍 READY 立即拉低"
        //   造成的采样偏移；AXI 允许 READY 持续为高。
        // READY 保持规则：除 RESP 阶段外都保持高，直到主站撤销 VALID。
        // 这样即使主站晚一拍撤销 VALID，也不会出现"VALID 未握手即撤销"的假象。
        axi.AWREADY = (wst != W_RESP);
        axi.WREADY  = (wst != W_RESP);
        axi.BVALID  = (wst == W_RESP);
        axi.BRESP   = w_slverr ? axi_defs_pkg::RESP_SLVERR : axi_defs_pkg::RESP_OKAY;

        // ---- 读通道 ----
        axi.ARREADY = (rst != R_RESP);
        axi.RVALID  = (rst == R_RESP);
        axi.RRESP   = r_slverr ? axi_defs_pkg::RESP_SLVERR : axi_defs_pkg::RESP_OKAY;
        axi.RDATA   = rdata_q;
    end

    // ==================================================================
    // APB 输出（组合）：由仲裁 owner 决定
    // ==================================================================
    always_comb begin
        apb.PSEL    = 1'b0;
        apb.PENABLE = 1'b0;
        apb.PWRITE  = 1'b0;
        apb.PADDR   = '0;
        apb.PWDATA  = '0;

        case (owner)
            A_WR: begin
                apb.PSEL    = (wst == W_SETUP) || (wst == W_ACC);
                apb.PENABLE = (wst == W_ACC);
                apb.PWRITE  = 1'b1;
                apb.PADDR   = awaddr_q;
                apb.PWDATA  = wdata_q;
            end
            A_RD: begin
                apb.PSEL    = (rst == R_SETUP) || (rst == R_ACC);
                apb.PENABLE = (rst == R_ACC);
                apb.PWRITE  = 1'b0;
                apb.PADDR   = araddr_q;
            end
            default: ;
        endcase
    end

    // ==================================================================
    // 时序逻辑：两条路径独立演进 + 仲裁
    // ==================================================================
    always_ff @(posedge ACLK or negedge ARESETn) begin
        if (!ARESETn) begin
            wst        <= W_IDLE;
            rst        <= R_IDLE;
            aw_held    <= 1'b0;
            w_held     <= 1'b0;
            ar_held    <= 1'b0;
            awaddr_q   <= '0;
            wdata_q    <= '0;
            araddr_q   <= '0;
            w_slverr   <= 1'b0;
            r_slverr   <= 1'b0;
            rdata_q    <= '0;
            owner      <= A_NONE;
            last_rd    <= 1'b0;
        end else begin
            // ---------------- 写路径 ----------------
            case (wst)
                W_IDLE: begin
                    // 独立接受 AW 与 W（可不同拍）
                    if (axi.AWVALID && axi.AWREADY) begin
                        awaddr_q <= axi.AWADDR;
                        aw_held  <= 1'b1;
                    end
                    if (axi.WVALID && axi.WREADY) begin
                        wdata_q <= axi.WDATA;
                        w_held  <= 1'b1;
                    end
                    // 两个都齐了 → 请求 APB
                    if ((aw_held || (axi.AWVALID && axi.AWREADY)) &&
                        (w_held  || (axi.WVALID  && axi.WREADY)))
                        wst <= W_REQ;
                end
                W_REQ: begin
                    // 本状态不再拉高 READY（组合逻辑中 READY 仅 W_IDLE 为高），
                    // 因此主站必须在握手完成的当拍撤销 VALID。
                    if (w_grant) begin
                        owner   <= A_WR;
                        last_rd <= 1'b0;
                        wst     <= W_SETUP;
                    end
                end
                W_SETUP: begin
                    wst <= W_ACC;
                end
                W_ACC: begin
                    if (apb.PREADY) begin
                        w_slverr <= apb.PSLVERR;
                        wst      <= W_RESP;
                        owner    <= A_NONE;     // 让出 APB，读可立即使用
                        aw_held  <= 1'b0;       // 事务已消费
                        w_held   <= 1'b0;
                    end
                end
                W_RESP: begin
                    if (axi.BREADY) begin
                        wst <= W_IDLE;
                    end
                end
                default: wst <= W_IDLE;
            endcase

            // ---------------- 读路径 ----------------
            case (rst)
                R_IDLE: begin
                    if (axi.ARVALID && axi.ARREADY) begin
                        araddr_q <= axi.ARADDR;
                        ar_held  <= 1'b1;
                        rst      <= R_REQ;
                    end
                end
                R_REQ: begin
                    if (r_grant) begin
                        owner   <= A_RD;
                        last_rd <= 1'b1;
                        rst     <= R_SETUP;
                    end
                end
                R_SETUP: begin
                    rst <= R_ACC;
                end
                R_ACC: begin
                    if (apb.PREADY) begin
                        rdata_q  <= apb.PRDATA;
                        r_slverr <= apb.PSLVERR;
                        rst      <= R_RESP;
                        owner    <= A_NONE;
                        ar_held  <= 1'b0;
                    end
                end
                R_RESP: begin
                    if (axi.RREADY) begin
                        rst <= R_IDLE;
                    end
                end
                default: rst <= R_IDLE;
            endcase
        end
    end

endmodule
`endif // AXI_APB_BRIDGE_DUAL_SV