// ============================================================================
// tb_gaps_sv：覆盖 TEST_PLAN 中 12 项缺口的场景测试套件
// ============================================================================
// 每项独立 task，互不干扰；全部通过则打印 PASS。
// 与主 TB（tb_axi_apb_sv）互补：主 TB 做随机回归，本文件做定向边界场景。
//
// 对应 TEST_PLAN.html 的缺口：
//   F-05 覆盖写同一地址      F-06 读未写地址初值
//   E-03 错误后恢复正常      E-04 多地址越界扫描
//   S-02 pause=0 背靠背      S-03 长 pause(>10)
//   R-01 上电复位后清零      R-02 事务中途复位
//   W-02 零等待              W-03 长等待
//   T-02 长时间随机          T-03 多 seed 回归
//
// 用法（参数化回归）：
//   仿真器 -GWAIT=0  -GSEED=7 ...   # W-02
//   仿真器 -GWAIT=16 -GSEED=7 ...   # W-03
//   仿真器 -GSEED=123 ...           # T-03 不同 seed
// ============================================================================
`ifndef TB_GAPS_SV_SV
`define TB_GAPS_SV_SV
`timescale 1ns / 1ps

module tb_gaps_sv #(
    parameter int AW   = 8,
    parameter int DW   = 32,
    parameter int NREG = 16,
    parameter int WAIT_CYCLES = 2,   // W-02/W-03 用 -G 覆盖
    parameter int SEED = 0           // T-03 用 -G 覆盖
);

    import bus_txn_pkg::*;
    import axi_defs_pkg::*;

    localparam int LIMIT = NREG * 4;

    // ---- 时钟/复位 ----
    logic ACLK = 0;
    logic ARESETn = 0;
    always #5 ACLK = ~ACLK;

    // ---- 接口与 DUT ----
    axi_lite_if #(.AW(AW), .DW(DW)) axi(ACLK);
    apb_if      #(.AW(AW), .DW(DW)) apb(ACLK);

    axi_apb_bridge_sv #(.AW(AW), .DW(DW)) bridge (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
    );
    apb_slave_regs_sv #(.AW(AW), .DW(DW), .NREG(NREG),
                        .WAIT_CYCLES(WAIT_CYCLES)) slave (
        .PCLK(ACLK), .PRESETn(ARESETn), .apb(apb)
    );

    axi_txn_host #(.AW(AW), .DW(DW)) host (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi)
    );

    // ---- 统计 ----
    int errors = 0;
    int checks = 0;

    function automatic txn_t txn(input bit wr, input [AW-1:0] a,
                                 input [DW-1:0] d, input [7:0] pause,
                                 input bit same, input bit exp_err);
        txn_t t;
        t = '0;
        t.write = wr; t.addr = a; t.data = d;
        t.pause = pause; t.same = same; t.exp_err = exp_err;
        return t;
    endfunction

    task automatic do_txn(input txn_t t, output rsp_t r);
        host.execute(t, r);
        checks++;
    endtask

    task automatic expect_ok(input txn_t t, input rsp_t r, input string tag);
        if (r.resp !== RESP_OKAY) begin
            $display("  [%0s] FAIL: resp=%s expect OKAY (addr=%0h)",
                     tag, resp_name(r.resp), t.addr);
            errors++;
        end
    endtask

    task automatic expect_slverr(input txn_t t, input rsp_t r, input string tag);
        if (r.resp !== RESP_SLVERR) begin
            $display("  [%0s] FAIL: resp=%s expect SLVERR (addr=%0h)",
                     tag, resp_name(r.resp), t.addr);
            errors++;
        end
    endtask

    task automatic do_reset();
        ARESETn = 0;
        repeat (4) @(posedge ACLK);
        ARESETn = 1;
    endtask

    // ==================================================================
    // F-05 覆盖写同一地址：连续写不同值，读回应为最后值
    // ==================================================================
    task automatic t_f05_overwrite();
        txn_t t; rsp_t r;
        $display("== F-05 覆盖写同一地址 ==");
        for (int i = 0; i < 5; i++) begin
            t = txn(1'b1, 8'h00, 32'hA000_0000 + i, 8'd0, 1'b0, 1'b0);
            do_txn(t, r); expect_ok(t, r, "F-05");
        end
        t = txn(1'b0, 8'h00, '0, 8'd0, 1'b0, 1'b0);
        do_txn(t, r);
        if (r.resp !== RESP_OKAY || r.data !== 32'hA000_0004) begin
            $display("  [F-05] FAIL: readback=%h expect A0000004", r.data);
            errors++;
        end
    endtask

    // ==================================================================
    // F-06 读未写地址初值：复位后读未访问地址应为 0
    // ==================================================================
    task automatic t_f06_unwritten();
        txn_t t; rsp_t r;
        $display("== F-06 读未写地址（初值）==");
        do_reset();
        // 先写一个地址，确保有其他内容，再读另一地址
        t = txn(1'b1, 8'h00, 32'hFFFF_FFFF, 8'd0, 1'b0, 1'b0);
        do_txn(t, r); expect_ok(t, r, "F-06");
        t = txn(1'b0, 8'h04, '0, 8'd0, 1'b0, 1'b0);
        do_txn(t, r);
        if (r.resp !== RESP_OKAY || r.data !== 32'h0) begin
            $display("  [F-06] FAIL: unwritten readback=%h expect 0", r.data);
            errors++;
        end
    endtask

    // ==================================================================
    // E-03 错误后恢复正常：越界访问后紧跟正常写读
    // ==================================================================
    task automatic t_e03_recover();
        txn_t t; rsp_t r;
        $display("== E-03 错误后恢复正常事务 ==");
        // 越界写 → SLVERR
        t = txn(1'b1, LIMIT[AW-1:0], 32'hBAD_0BAD, 8'd0, 1'b0, 1'b1);
        do_txn(t, r); expect_slverr(t, r, "E-03-oob-wr");
        // 紧跟正常写
        t = txn(1'b1, 8'h08, 32'h1234_5678, 8'd0, 1'b0, 1'b0);
        do_txn(t, r); expect_ok(t, r, "E-03-ok-wr");
        // 读回应正确（验证 slv_err 已清除）
        t = txn(1'b0, 8'h08, '0, 8'd0, 1'b0, 1'b0);
        do_txn(t, r);
        if (r.resp !== RESP_OKAY || r.data !== 32'h1234_5678) begin
            $display("  [E-03] FAIL: post-error readback=%h resp=%s",
                     r.data, resp_name(r.resp));
            errors++;
        end
        // 越界读 → SLVERR
        t = txn(1'b0, LIMIT[AW-1:0], '0, 8'd0, 1'b0, 1'b1);
        do_txn(t, r); expect_slverr(t, r, "E-03-oob-rd");
        // 再正常读
        t = txn(1'b0, 8'h08, '0, 8'd0, 1'b0, 1'b0);
        do_txn(t, r);
        if (r.resp !== RESP_OKAY) begin
            $display("  [E-03] FAIL: after oob-rd resp=%s", resp_name(r.resp));
            errors++;
        end
    endtask

    // ==================================================================
    // E-04 多地址越界扫描
    // ==================================================================
    task automatic t_e04_oob_sweep();
        txn_t t; rsp_t r;
        $display("== E-04 多地址越界扫描 ==");
        for (int k = 0; k < 8; k++) begin
            automatic bit [AW-1:0] a = LIMIT + k*4;
            t = txn(1'b1, a, 32'h0, 8'd0, 1'b0, 1'b1);
            do_txn(t, r); expect_slverr(t, r, "E-04-wr");
            t = txn(1'b0, a, '0, 8'd0, 1'b0, 1'b1);
            do_txn(t, r); expect_slverr(t, r, "E-04-rd");
        end
        // 越界写不应污染最后一个合法寄存器
        t = txn(1'b1, LIMIT-4, 32'hCAFE_1234, 8'd0, 1'b0, 1'b0);
        do_txn(t, r); expect_ok(t, r, "E-04-last-wr");
        t = txn(1'b1, LIMIT, 32'hDEAD_DEAD, 8'd0, 1'b0, 1'b1);
        do_txn(t, r); expect_slverr(t, r, "E-04-oob-wr");
        t = txn(1'b0, LIMIT-4, '0, 8'd0, 1'b0, 1'b0);
        do_txn(t, r);
        if (r.data !== 32'hCAFE_1234) begin
            $display("  [E-04] FAIL: last reg corrupted: %h", r.data);
            errors++;
        end
    endtask

    // ==================================================================
    // S-02 pause=0 背靠背（验证无死锁、数据不串）
    // ==================================================================
    task automatic t_s02_backtoback();
        txn_t t; rsp_t r;
        $display("== S-02 pause=0 背靠背 ==");
        for (int i = 0; i < 8; i++) begin
            t = txn(1'b1, i[AW-1:0]*4, 32'hB0B0_0000 + i, 8'd0, 1'b0, 1'b0);
            do_txn(t, r); expect_ok(t, r, "S-02-wr");
        end
        for (int i = 0; i < 8; i++) begin
            t = txn(1'b0, i[AW-1:0]*4, '0, 8'd0, 1'b0, 1'b0);
            do_txn(t, r);
            if (r.data !== 32'hB0B0_0000 + i) begin
                $display("  [S-02] FAIL: addr=%0d data=%h expect %h",
                         i, r.data, 32'hB0B0_0000 + i);
                errors++;
            end
        end
        // pause=0 时 r.idle 应为 0
        t = txn(1'b0, 8'h00, '0, 8'd0, 1'b0, 1'b0);
        do_txn(t, r);
        if (r.idle !== 0) begin
            $display("  [S-02] FAIL: pause=0 but idle=%0d", r.idle);
            errors++;
        end
    endtask

    // ==================================================================
    // S-03 长 pause（>10 拍）
    // ==================================================================
    task automatic t_s03_long_pause();
        txn_t t; rsp_t r;
        $display("== S-03 长 pause ==");
        t = txn(1'b1, 8'h0C, 32'h5555_AAAA, 8'd20, 1'b0, 1'b0);
        do_txn(t, r); expect_ok(t, r, "S-03-wr");
        if (r.idle !== 20) begin
            $display("  [S-03] FAIL: pause=20 but idle=%0d", r.idle);
            errors++;
        end
        t = txn(1'b0, 8'h0C, '0, 8'd16, 1'b0, 1'b0);
        do_txn(t, r);
        if (r.data !== 32'h5555_AAAA) begin
            $display("  [S-03] FAIL: long-pause readback=%h", r.data);
            errors++;
        end
        if (r.idle !== 16) begin
            $display("  [S-03] FAIL: pause=16 but idle=%0d", r.idle);
            errors++;
        end
    endtask

    // ==================================================================
    // R-01 上电复位后寄存器清零
    // ==================================================================
    task automatic t_r01_reset_clear();
        txn_t t; rsp_t r;
        $display("== R-01 上电复位后寄存器清零 ==");
        // 先写脏
        for (int i = 0; i < NREG; i++) begin
            t = txn(1'b1, i[AW-1:0]*4, 32'hDEAD_0000 + i, 8'd0, 1'b0, 1'b0);
            do_txn(t, r);
        end
        // 复位
        do_reset();
        // 全部应为 0
        for (int i = 0; i < NREG; i++) begin
            t = txn(1'b0, i[AW-1:0]*4, '0, 8'd0, 1'b0, 1'b0);
            do_txn(t, r);
            if (r.resp !== RESP_OKAY || r.data !== 0) begin
                $display("  [R-01] FAIL: addr=%0d after reset=%h resp=%s",
                         i, r.data, resp_name(r.resp));
                errors++;
            end
        end
    endtask

    // ==================================================================
    // R-02 事务中途复位
    // ==================================================================
    // 机制：在独立线程发起一笔"分开发"写（AW 后 W 前有间隙），
    //       主线程此时拉低 ARESETn 复位，验证复位后总线回归 IDLE、DUT 可继续工作。
    task automatic t_r02_mid_reset();
        txn_t t; rsp_t r;
        $display("== R-02 事务中途复位 ==");

        // 启动一个后台写事务（会卡在 AW/W 握手等待）
        fork
            begin
                txn_t tb = txn(1'b1, 8'h10, 32'hAAAA_1111, 8'd30, 1'b0, 1'b0);
                rsp_t rb;
                host.execute(tb, rb);
            end
        join_none

        // 等几拍让事务进行到一半
        repeat (3) @(posedge ACLK);
        // 中途复位
        ARESETn = 0;
        repeat (4) @(posedge ACLK);
        ARESETn = 1;
        // 等后台事务收尾（复位会打断它）
        repeat (20) @(posedge ACLK);

        // 复位后首先验证总线能正常工作
        t = txn(1'b1, 8'h10, 32'h1234_ABAB, 8'd0, 1'b0, 1'b0);
        do_txn(t, r); expect_ok(t, r, "R-02-post-wr");
        t = txn(1'b0, 8'h10, '0, 8'd0, 1'b0, 1'b0);
        do_txn(t, r);
        if (r.resp !== RESP_OKAY || r.data !== 32'h1234_ABAB) begin
            $display("  [R-02] FAIL: post-reset txn broken: data=%h resp=%s",
                     r.data, resp_name(r.resp));
            errors++;
        end
    endtask

    // ==================================================================
    // T-02 / T-03：长随机 + seed 回归（由 -GSEED 控制 seed）
    // ==================================================================
    task automatic t_t02_long_random(input int n);
        txn_t t; rsp_t r;
        bit [DW-1:0] mirror [0:NREG-1];
        int idx;
        $display("== T-02 长随机（%0d 笔, WAIT=%0d, SEED=%0d）==", n, WAIT_CYCLES, SEED);
        // 清零镜像
        for (int i = 0; i < NREG; i++) mirror[i] = '0;
        do_reset();

        for (int k = 0; k < n; k++) begin
            automatic bit wr = $urandom(SEED + k) & 1;
            automatic bit [AW-1:0] a = $urandom(SEED + k + 1) % LIMIT;
            automatic bit [DW-1:0] d = $urandom(SEED + k + 2);
            automatic bit [7:0] pz = $urandom(SEED + k + 3) % 3;
            idx = a >> 2;
            t = txn(wr, a, d, pz, 1'b0, 1'b0);
            do_txn(t, r);
            if (r.resp !== RESP_OKAY) begin
                $display("  [T-02] FAIL k=%0d: resp=%s", k, resp_name(r.resp));
                errors++;
            end else if (wr) begin
                mirror[idx] = d;
            end else if (r.data !== mirror[idx]) begin
                $display("  [T-02] FAIL k=%0d addr=%0h exp=%h got=%h",
                         k, a, mirror[idx], r.data);
                errors++;
            end
        end
    endtask

    // ==================================================================
    // 主流程
    // ==================================================================
    initial begin
        $display("======================================================");
        $display(" gaps suite: WAIT_CYCLES=%0d SEED=%0d NREG=%0d",
                 WAIT_CYCLES, SEED, NREG);
        $display("======================================================");

        do_reset();

        t_f05_overwrite();
        t_f06_unwritten();
        t_e03_recover();
        t_e04_oob_sweep();
        t_s02_backtoback();
        t_s03_long_pause();
        t_r01_reset_clear();
        t_r02_mid_reset();
        t_t02_long_random(200);   // T-02/T-03：种子由 -GSEED 控制

        $display("------------------------------------------------------");
        $display(" gaps result: %0d checks, %0d errors (WAIT=%0d SEED=%0d)",
                 checks, errors, WAIT_CYCLES, SEED);
        if (errors == 0) begin
            $display("PASS: gaps suite (WAIT=%0d SEED=%0d)", WAIT_CYCLES, SEED);
            $finish;
        end else begin
            $display("FAIL: %0d errors", errors);
            $fatal(1, "gaps suite failed");
        end
    end

    initial begin
        $dumpfile("tb_gaps_sv.vcd");
        $dumpvars(0, tb_gaps_sv);
    end

endmodule
`endif // TB_GAPS_SV_SV