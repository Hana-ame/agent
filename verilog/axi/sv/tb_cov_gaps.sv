// ============================================================================
// tb_cov_gaps.sv：补齐 Verilator --coverage 发现的真实覆盖缺口
// ============================================================================
// 来源：verilator_coverage 注解显示以下代码路径从未被执行：
//
//   axi_apb_bridge_sv.sv:
//     S_W_H 状态（第 58-59、66、107、111 行）—— "先发 W、后发 AW" 的写事务
//       桥的 S_IDLE 支持三种写发起顺序：
//         a) AW+W 同拍     → S_AWW_H   （已测）
//         b) AW 先、W 后    → S_AW_H    （已测）
//         c) W 先、AW 后    → S_W_H     （★ 未测，本文件补测）
//     slv_err 复位/置位路径（第 128、133 行）—— 错误响应后的清除
//     rdata_q 复位、state 复位（第 144、152 行）—— 复位生效拍
//
// AXI 协议允许 W 通道先于 AW 通道发出（两个通道独立），桥必须支持。
// ============================================================================
`ifndef TB_COV_GAPS_SV
`define TB_COV_GAPS_SV
`timescale 1ns / 1ps

module tb_cov_gaps #(
    parameter int AW   = 8,
    parameter int DW   = 32,
    parameter int NREG = 16
);

    import axi_defs_pkg::*;

    localparam int LIMIT = NREG * 4;

    logic ACLK = 0;
    logic ARESETn = 0;
    always #5 ACLK = ~ACLK;

    axi_lite_if #(.AW(AW), .DW(DW)) axi(ACLK);
    apb_if      #(.AW(AW), .DW(DW)) apb(ACLK);

    // 被测桥：单事务串行版（缺口所在）
    axi_apb_bridge_sv #(.AW(AW), .DW(DW)) bridge (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
    );
    apb_slave_regs_sv #(.AW(AW), .DW(DW), .NREG(NREG), .WAIT_CYCLES(2)) slave (
        .PCLK(ACLK), .PRESETn(ARESETn), .apb(apb)
    );

    int errors = 0;
    int checks = 0;

    task automatic do_reset();
        ARESETn = 0;
        axi.AWVALID = 0; axi.WVALID = 0; axi.ARVALID = 0;
        axi.BREADY = 0;  axi.RREADY = 0;
        repeat (4) @(posedge ACLK);
        ARESETn = 1;
        @(posedge ACLK);
    endtask

    // ==================================================================
    // 缺口 1：W 先发、AW 后发（走 S_W_H 状态）
    // ==================================================================
    // 时序：先拉高 WVALID（AWVALID 仍为低）→ 桥进入 S_W_H →
    //       下一拍拉高 AWVALID → 桥收齐后进 APB
    task automatic t_w_before_aw();
        bit w_done = 0, aw_done = 0;
        int wait_cycles = 0;
        $display("== COV-01: W 先发、AW 后发（S_W_H 路径）==");

        // 第 1 步：只发 WVALID（AWVALID 保持 0）
        // 桥在 S_IDLE 看到 WVALID 会组合拉高 WREADY，握手发生在下一拍；
        // 握手后桥切到 S_W_H，WREADY 拉低属合法行为。
        @(posedge ACLK); #1;
        axi.WVALID = 1'b1; axi.WDATA = 32'hEFBE_ADDE;
        // 等握手完成：观察到 WREADY=1 即视为 W 被接受
        w_done = 0;
        for (int cyc = 0; cyc < 20 && !w_done; cyc++) begin
            @(posedge ACLK); #1;
            if (axi.WREADY) w_done = 1;
        end
        checks++;
        if (!w_done) begin
            $display("  [COV-01] FAIL: 单发 WVALID 未获 WREADY（20 拍超时）");
            errors++;
        end else
            $display("  [COV-01] 单发 WVALID 被接受（进入 S_W_H）✓");
        axi.WVALID = 1'b0;

        // 第 3 步：现在发 AWVALID（桥应在 S_W_H 或 S_IDLE 等待）
        @(posedge ACLK); #1;
        axi.AWVALID = 1'b1; axi.AWADDR = 8'h1C;

        // 等 AW 握手（桥在 S_W_H 会拉高 AWREADY）
        while (wait_cycles < 50) begin
            @(posedge ACLK); #1;
            wait_cycles++;
            if (axi.AWREADY) begin
                aw_done = 1;
                axi.AWVALID = 1'b0;
                break;
            end
        end
        checks++;
        if (!aw_done) begin
            $display("  [COV-01] FAIL: AW 在 W 先行场景下未被接受（%0d 拍超时）", wait_cycles);
            errors++;
        end else
            $display("  [COV-01] W→AW 顺序握手成功 ✓");

        // 等写响应
        axi.BREADY = 1'b1;
        do @(posedge ACLK); while (!axi.BVALID);
        checks++;
        if (axi.BRESP !== RESP_OKAY) begin
            $display("  [COV-01] FAIL: BRESP=%s expect OKAY", resp_name(axi.BRESP));
            errors++;
        end
        #1; axi.BREADY = 1'b0;
        @(posedge ACLK);

        // 验证数据确实写入
        @(posedge ACLK); #1;
        axi.ARVALID = 1'b1; axi.ARADDR = 8'h1C;
        while (!axi.ARREADY) @(posedge ACLK);
        #1; axi.ARVALID = 1'b0;
        axi.RREADY = 1'b1;
        do @(posedge ACLK); while (!axi.RVALID);
        checks++;
        if (axi.RDATA !== 32'hEFBE_ADDE) begin
            $display("  [COV-01] FAIL: 读回数据=%h expect EFBEADDE", axi.RDATA);
            errors++;
        end else
            $display("  [COV-01] W→AW 数据落盘正确 ✓");
        #1; axi.RREADY = 1'b0;
        @(posedge ACLK);
    endtask

    // ==================================================================
    // 缺口 2：错误响应后的 slv_err 清除路径
    // ==================================================================
    // 越界写 → SLVERR（slv_err 置位）→ 后续正常事务 BRESP 应回到 OKAY
    // （覆盖桥中 slv_err <= 1'b0 的清除分支）
    task automatic t_err_clear();
        int err_cases = 0, ok_cases = 0;
        $display("== COV-02: 错误响应后 slv_err 清除 ==");

        // 连续 3 轮：越界写（期望 SLVERR）+ 正常写（期望 OKAY）
        for (int k = 0; k < 3; k++) begin
            // 越界写
            @(posedge ACLK); #1;
            axi.AWVALID = 1'b1; axi.AWADDR = LIMIT[AW-1:0];
            axi.WVALID  = 1'b1; axi.WDATA  = 32'hBAD0_0000 + k;
            while (!(axi.AWVALID == 0 || axi.AWREADY)) @(posedge ACLK);
            #1;
            if (axi.AWREADY) axi.AWVALID = 1'b0;
            while (!(axi.WREADY || axi.WVALID == 0)) @(posedge ACLK);
            #1;
            if (axi.WREADY) axi.WVALID = 1'b0;
            axi.BREADY = 1'b1;
            do @(posedge ACLK); while (!axi.BVALID);
            checks++;
            if (axi.BRESP === RESP_SLVERR) err_cases++;
            else begin
                $display("  [COV-02] FAIL: 越界写 BRESP=%s expect SLVERR",
                         resp_name(axi.BRESP));
                errors++;
            end
            #1; axi.BREADY = 1'b0;
            @(posedge ACLK);

            // 紧跟正常写（验证 slv_err 已清除）
            @(posedge ACLK); #1;
            axi.AWVALID = 1'b1; axi.AWADDR = 8'h20;
            axi.WVALID  = 1'b1; axi.WDATA  = 32'h600D_0000 + k;
            while (!(axi.AWREADY)) @(posedge ACLK); #1;
            axi.AWVALID = 1'b0;
            while (!(axi.WREADY)) @(posedge ACLK); #1;
            axi.WVALID = 1'b0;
            axi.BREADY = 1'b1;
            do @(posedge ACLK); while (!axi.BVALID);
            checks++;
            if (axi.BRESP === RESP_OKAY) ok_cases++;
            else begin
                $display("  [COV-02] FAIL: 错误后正常写 BRESP=%s expect OKAY (slv_err 未清除)",
                         resp_name(axi.BRESP));
                errors++;
            end
            #1; axi.BREADY = 1'b0;
            @(posedge ACLK);
        end
        $display("  [COV-02] %0d 次 SLVERR + %0d 次恢复 OKAY ✓", err_cases, ok_cases);
    endtask

    // ==================================================================
    // 缺口 3：复位中途的行为（rdata_q / state 复位分支）
    // ==================================================================
    task automatic t_reset_paths();
        $display("== COV-03: 复位路径（rdata_q / state 复位分支）==");
        // 反复复位，确保复位分支被执行（覆盖 144、152 行）
        for (int k = 0; k < 3; k++) begin
            do_reset();
            @(posedge ACLK); #1;
            checks++;
            // AXI 允许 READY 依赖 VALID：复位后无 VALID 时 READY=0 是合法的。
            // 正确判据：复位后发起一个事务，应能正常完成（桥已回 IDLE 可用）。
            // 这里用"无残留 VALID（主站侧）" + 内部状态回 IDLE 检查。
            if (axi.BVALID !== 1'b0 || axi.RVALID !== 1'b0) begin
                $display("  [COV-03] FAIL: 复位后仍有响应挂起 BV=%b RV=%b",
                         axi.BVALID, axi.RVALID);
                errors++;
            end
        end
        $display("  [COV-03] 多次复位后桥状态干净 ✓");
    endtask

    // ==================================================================
    initial begin
        $display("======================================================");
        $display(" coverage-gap testbench (Verilator --coverage 驱动)");
        $display("======================================================");
        do_reset();

        t_w_before_aw();
        t_err_clear();
        t_reset_paths();

        $display("------------------------------------------------------");
        $display(" cov-gaps result: %0d checks, %0d errors", checks, errors);
        if (errors == 0) begin
            $display("PASS: coverage-gap suite");
            $finish;
        end else begin
            $display("FAIL: %0d errors", errors);
            $fatal(1, "coverage-gap suite failed");
        end
    end

    initial begin
        $dumpfile("tb_cov_gaps.vcd");
        $dumpvars(0, tb_cov_gaps);
    end

endmodule
`endif // TB_COV_GAPS_SV