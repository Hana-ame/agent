// 完整 SystemVerilog testbench：AXI4-Lite → APB → 慢速外设 → 响应回 AXI
//
// ============================================================================
// 分层：定义简单到就是结构体
// ============================================================================
//   第 3 层（本文件）  txn_t / rsp_t —— 只是两个 struct，一眼读完
//                       本文件里【不出现任何 AXI 信号名】：没有 AWVALID、
//                       没有 WREADY、没有 clocking —— 只要写
//                           txn_t t = txn(1'b1, 8'h00, 32'hDEAD, 8'd3, 1'b0, 1'b0);
//                           host.execute(t, r);
//      ↓
//   第 2 层（axi_txn_host.sv）  把 txn_t 展开成 AXI 握手时序，pause → 空闲拍
//      ↓
//   第 1 层（axi_lite_if / bridge / apb_if / slave）  协议细节
//
// ============================================================================
// 测试闭环：数据向量 → 激励向量 → 数据向量
// ============================================================================
//   ① 数据向量集：txn_t 队列（含 pause / same / exp_err 控制字段）
//   ② 激励向量 ：host.execute(t, r) 逐条展开成总线时序
//   ③ 数据向量 ：r vs 期望（参考镜像 ref_mem）比对；pause 可观测回读
`ifndef TB_AXI_APB_SV_SV
`define TB_AXI_APB_SV_SV
`timescale 1ns / 1ps

module tb_axi_apb_sv;

    import bus_txn_pkg::*;
    import axi_defs_pkg::*;

    localparam int AW   = 8;
    localparam int DW   = 32;
    localparam int NREG = 16;
    localparam int WAIT_CYCLES = 2;
    localparam int NTXN = 200;   // 随机事务数

    // ---- 时钟/复位 ----
    logic ACLK = 0;
    logic ARESETn = 0;
    always #5 ACLK = ~ACLK;

    // ---- 接口与 DUT（第 1 层，本文件不直接操作这些信号）----
    axi_lite_if #(.AW(AW), .DW(DW)) axi(ACLK);
    apb_if      #(.AW(AW), .DW(DW)) apb(ACLK);

    // 独立协议监视器（被动只读，自动检查 AXI/APB 合法性）
    axi_apb_monitor #(.AW(AW), .DW(DW), .NAME("SV")) mon (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
    );

    axi_apb_bridge_sv #(.AW(AW), .DW(DW)) bridge (
        .ACLK   (ACLK),
        .ARESETn(ARESETn),
        .axi    (axi),
        .apb    (apb)
    );

    apb_slave_regs_sv #(.AW(AW), .DW(DW), .NREG(NREG),
                        .WAIT_CYCLES(WAIT_CYCLES)) slave (
        .PCLK   (ACLK),
        .PRESETn(ARESETn),
        .apb    (apb)
    );

    // ---- 第 2 层：事务适配器（上层只用它的 execute）----
    axi_txn_host #(.AW(AW), .DW(DW)) host (
        .ACLK   (ACLK),
        .ARESETn(ARESETn),
        .axi    (axi)
    );

    // ========================================================================
    // ① 数据向量集：就是 txn_t 的队列（可读、可审、可重放）
    // ========================================================================
    txn_t stim_vecs[$];
    logic [DW-1:0] ref_mem [0:NREG-1];

    // 构造一条事务向量
    function automatic txn_t txn(input bit wr, input [AW-1:0] a,
                                 input [DW-1:0] d, input [7:0] pause,
                                 input bit same, input bit exp_err);
        txn_t t;
        t = '0;
        t.write   = wr;
        t.addr    = a;
        t.data    = d;
        t.pause   = pause;
        t.same    = same;
        t.exp_err = exp_err;
        return t;
    endfunction

    int vec_cnt = 0;
    initial begin
        // ①a 回环基本集：写满 → 读回（每笔写后 pause 1 拍）
        for (int i = 0; i < NREG; i++)
            stim_vecs.push_back(txn(1'b1, i[AW-1:0]*4, $urandom, 8'd1, 1'b0, 1'b0));
        for (int i = 0; i < NREG; i++)
            stim_vecs.push_back(txn(1'b0, i[AW-1:0]*4, '0, 8'd0, 1'b0, 1'b0));
        // ①b 随机向量：写/读混合，随机 pause(0~3) 与 same
        for (int k = 0; k < NTXN; k++) begin
            automatic bit wr   = $urandom & 1;
            automatic bit same = wr ? ($urandom & 1) : 1'b0;
            automatic bit [7:0] pz = $urandom % 4;
            automatic bit [AW-1:0] a = $urandom % (NREG*4);
            stim_vecs.push_back(txn(wr, a, $urandom, pz, same, 1'b0));
        end
        // ①c 边界向量：越界读写（exp_err=1，期望 SLVERR）
        stim_vecs.push_back(txn(1'b1, NREG*4, 32'hDEAD_BEEF, 8'd2, 1'b0, 1'b1));
        stim_vecs.push_back(txn(1'b0, NREG*4, '0, 8'd2, 1'b0, 1'b1));
        vec_cnt = stim_vecs.size();
        $display("== 1 data vectors: %0d built (loopback %0d + random %0d + oob 2) ==",
                 vec_cnt, NREG*2, NTXN);
    end

    // ========================================================================
    // ③ 数据向量比对：只认识 txn_t / rsp_t 的字段
    // ========================================================================
    int errors = 0;
    int checks = 0;
    int pause_checks = 0;

    function automatic void check_rsp(input txn_t t, input rsp_t r);
        int idx = t.addr >> 2;
        checks++;
        // 期望错误（越界）→ 实际必须是错误响应
        if (t.exp_err) begin
            if (r.resp === RESP_OKAY) begin
                $display("SB FAIL[%0d]: %s addr=%0d expect SLVERR got OKAY",
                         checks, t.write ? "WR" : "RD", t.addr);
                errors++;
            end
        end else begin
            if (r.resp !== RESP_OKAY) begin
                $display("SB FAIL[%0d]: %s addr=%0d expect OKAY got %s",
                         checks, t.write ? "WR" : "RD", t.addr, resp_name(r.resp));
                errors++;
            end else if (t.write) begin
                ref_mem[idx] = t.data;          // 写向量 → 更新期望镜像
            end else if (r.data !== ref_mem[idx]) begin
                $display("SB FAIL[%0d]: RD addr=%0d exp=%h got=%h",
                         checks, t.addr, ref_mem[idx], r.data);
                errors++;
            end
        end
        // pause 可观测性：发了几拍空闲就该回读几拍
        if (t.pause != 0) begin
            pause_checks++;
            if (r.idle !== t.pause) begin
                $display("SB FAIL[%0d]: pause=%0d but idle observed=%0d",
                         checks, t.pause, r.idle);
                errors++;
            end
        end
    endfunction

    // ========================================================================
    // ② 激励向量 → ③ 数据向量：主流程（这里只有 txn_t 和 rsp_t）
    // ========================================================================
    int txns = 0;
    int total_pause = 0;

    initial begin
        // 复位
        ARESETn = 0;
        repeat (4) @(posedge ACLK);
        ARESETn = 1;

        $display("== SV testbench: AXI->APB->AXI transaction layer (WAIT=%0d) ==",
                 WAIT_CYCLES);
        $display("== 2 stimulus: %0d txns via host.execute(txn_t) ==", vec_cnt);

        foreach (stim_vecs[i]) begin
            rsp_t r;
            host.execute(stim_vecs[i], r);   // ← 上层唯一需要写的调用
            check_rsp(stim_vecs[i], r);
            total_pause += stim_vecs[i].pause;
            txns++;
        end

        // ---- 汇总 ----
        $display("== 3 data compare: %0d checks, %0d pause checks, %0d errors ==",
                 checks, pause_checks, errors);
        $display("-- SUMMARY: %0d txns, %0d pause cycles injected --",
                 txns, total_pause);
        if (errors == 0) begin
            if (mon.errors != 0) begin
                $display("  [MON] FAIL: 协议监视器发现 %0d 处违规", mon.errors);
                errors += mon.errors;
            end else
                $display("  [MON] 协议监视器通过（观测 %0d 次 APB 传输）", mon.xfer_count);

            $display("PASS: axi->apb->axi transaction-layer testbench (%0d txns)", txns);
            $finish;
        end else begin
            $display("FAIL: %0d errors", errors);
            $fatal(1, "transaction-layer testbench failed");
        end
    end

    // ---- 波形 ----
    initial begin
        $dumpfile("tb_axi_apb_sv.vcd");
        $dumpvars(0, tb_axi_apb_sv);
    end

endmodule
`endif // TB_AXI_APB_SV_SV