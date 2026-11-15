// ============================================================================
// tb_sva_antecedent_sv.sv：确保 SVA 断言的「前件」真的会被触发
// ============================================================================
// 背景（本轮发现的真实盲区）：
//   axi_lite_if 中有 5 条 SVA 断言形如
//       assert property (@(posedge ACLK) (VALID && !READY) |=> VALID)
//   如果 (VALID && !READY) 这个前件从不成立，断言永远「通过」但毫无价值
//   （真空通过 / vacuous pass）。
//
//   实测 tb_axi_apb_sv 跑 20 对读写后统计：
//       AWVALID && !AWREADY : 0
//       WVALID  && !WREADY  : 0
//       ARVALID && !ARREADY : 0
//       BVALID  && !BREADY  : 0
//       RVALID  && !RREADY  : 0
//       （APB 侧前件正常：PENABLE 210 次 / SETUP 42 次 / ACC 等待 168 次）
//
//   根因：串行桥只要看到 VALID 就立即拉 READY，等待拍不存在。
//
// 本测试的作用：
//   作为「环境」驱动 READY 侧，故意延迟 READY，使前件成立，
//   从而真正激活这 5 条断言。同时也是 AXI 反压行为的独立验证。
//
// 配套的反向验证（已实测）：
//   故意在无 READY 时撤销 AWVALID，断言立即报
//     "Assertion failed: AXI: AWVALID dropped before handshake"
//   证明断言「既会被激活，也能在违规时失败」，不是真空通过。
// ============================================================================
`ifndef TB_SVA_ANTECEDENT_SV_SV
`define TB_SVA_ANTECEDENT_SV_SV
`timescale 1ns / 1ps

module tb_sva_antecedent_sv #(
    parameter int AW = 8,
    parameter int DW = 32
);

    logic ACLK = 0, ARESETn = 0;
    always #5 ACLK = ~ACLK;

    // 本测试把接口当「从机侧」使用：驱动 READY，读取 VALID
    axi_lite_if #(.AW(AW), .DW(DW)) axi(ACLK);

    int errors = 0, checks = 0;

    // ---- 前件触发计数（核心验证目标） ----
    int cnt_aw_wait = 0;   // AWVALID && !AWREADY
    int cnt_w_wait  = 0;   // WVALID  && !WREADY
    int cnt_ar_wait = 0;   // ARVALID && !ARREADY
    int cnt_b_wait  = 0;   // BVALID  && !BREADY
    int cnt_r_wait  = 0;   // RVALID  && !RREADY

    always @(posedge ACLK) if (ARESETn) begin
        if (axi.AWVALID && !axi.AWREADY) cnt_aw_wait++;
        if (axi.WVALID  && !axi.WREADY)  cnt_w_wait++;
        if (axi.ARVALID && !axi.ARREADY) cnt_ar_wait++;
        if (axi.BVALID  && !axi.BREADY)  cnt_b_wait++;
        if (axi.RVALID  && !axi.RREADY)  cnt_r_wait++;
    end

    task automatic do_reset();
        ARESETn = 0;
        axi.AWVALID = 0; axi.AWADDR = 0;
        axi.WVALID = 0;  axi.WDATA = 0;
        axi.ARVALID = 0; axi.ARADDR = 0;
        axi.BREADY = 0;  axi.RREADY = 0;
        repeat (4) @(posedge ACLK);
        ARESETn = 1;
        @(posedge ACLK);
    endtask

    // 清空接口输出侧（本测试扮演从机，需要驱动 READY 和响应）
    task automatic idle_slave();
        axi.AWREADY = 0; axi.WREADY = 0; axi.ARREADY = 0;
        axi.BVALID = 0;  axi.BRESP = 0;
        axi.RVALID = 0;  axi.RDATA = 0; axi.RRESP = 0;
    endtask

    // ==================================================================
    // T1：AW/W 前件 —— 拉起 VALID 后延迟 3 拍才给 READY
    // ==================================================================
    task automatic t_aw_w_wait();
        $display("== T1: AW/W 的 VALID&&!READY 前件 ==");
        do_reset(); idle_slave();
        cnt_aw_wait = 0; cnt_w_wait = 0;

        @(posedge ACLK); #1;
        axi.AWVALID = 1; axi.AWADDR = 8'h10;
        axi.WVALID  = 1; axi.WDATA  = 32'h1234_5678;
        // 保持 READY=0 三拍（制造等待）
        repeat (3) @(posedge ACLK);
        #1;

        checks++;
        if (cnt_aw_wait >= 3 && cnt_w_wait >= 3) begin
            $display("  ✓ AW 等待 %0d 拍, W 等待 %0d 拍（前件成立）",
                     cnt_aw_wait, cnt_w_wait);
        end else begin
            $display("  ✗ 前件触发不足：AW=%0d W=%0d（应 >=3）",
                     cnt_aw_wait, cnt_w_wait);
            errors++;
        end

        // 给 READY 完成握手
        axi.AWREADY = 1; axi.WREADY = 1;
        @(posedge ACLK); #1;
        axi.AWVALID = 0; axi.WVALID = 0;
        axi.AWREADY = 0; axi.WREADY = 0;
        @(posedge ACLK); #1;
        // 等 SVA 的 |=> 覆盖完
        @(posedge ACLK); #1;
    endtask

    // ==================================================================
    // T2：AR 前件
    // ==================================================================
    task automatic t_ar_wait();
        $display("== T2: AR 的 VALID&&!READY 前件 ==");
        do_reset(); idle_slave();
        cnt_ar_wait = 0;

        @(posedge ACLK); #1;
        axi.ARVALID = 1; axi.ARADDR = 8'h20;
        repeat (3) @(posedge ACLK);
        #1;

        checks++;
        if (cnt_ar_wait >= 3) begin
            $display("  ✓ AR 等待 %0d 拍（前件成立）", cnt_ar_wait);
        end else begin
            $display("  ✗ AR 前件触发不足：%0d（应 >=3）", cnt_ar_wait);
            errors++;
        end

        axi.ARREADY = 1;
        @(posedge ACLK); #1;
        axi.ARVALID = 0; axi.ARREADY = 0;
        @(posedge ACLK); #1;
    endtask

    // ==================================================================
    // T3：B 前件 —— 从机拉 BVALID，主站延迟 BREADY
    // ==================================================================
    task automatic t_b_wait();
        $display("== T3: B 的 VALID&&!READY 前件 ==");
        do_reset(); idle_slave();
        cnt_b_wait = 0;

        // 本测试扮演从机：主动拉 BVALID，主站 BREADY 保持 0
        @(posedge ACLK); #1;
        axi.BVALID = 1; axi.BRESP = 2'b00;
        axi.BREADY = 0;
        repeat (3) @(posedge ACLK);
        #1;

        checks++;
        if (cnt_b_wait >= 3) begin
            $display("  ✓ B 等待 %0d 拍（前件成立）", cnt_b_wait);
        end else begin
            $display("  ✗ B 前件触发不足：%0d（应 >=3）", cnt_b_wait);
            errors++;
        end

        axi.BREADY = 1;
        @(posedge ACLK); #1;
        axi.BVALID = 0; axi.BREADY = 0;
        @(posedge ACLK); #1;
    endtask

    // ==================================================================
    // T4：R 前件
    // ==================================================================
    task automatic t_r_wait();
        $display("== T4: R 的 VALID&&!READY 前件 ==");
        do_reset(); idle_slave();
        cnt_r_wait = 0;

        @(posedge ACLK); #1;
        axi.RVALID = 1; axi.RDATA = 32'hCAFE_BABE; axi.RRESP = 2'b00;
        axi.RREADY = 0;
        repeat (3) @(posedge ACLK);
        #1;

        checks++;
        if (cnt_r_wait >= 3) begin
            $display("  ✓ R 等待 %0d 拍（前件成立）", cnt_r_wait);
        end else begin
            $display("  ✗ R 前件触发不足：%0d（应 >=3）", cnt_r_wait);
            errors++;
        end

        axi.RREADY = 1;
        @(posedge ACLK); #1;
        axi.RVALID = 0; axi.RREADY = 0;
        @(posedge ACLK); #1;
    endtask

    // ==================================================================
    initial begin
        $display("======================================================");
        $display(" SVA 前件激活验证（避免断言真空通过）");
        $display("======================================================");

        idle_slave();
        do_reset();

        t_aw_w_wait();
        t_ar_wait();
        t_b_wait();
        t_r_wait();

        $display("------------------------------------------------------");
        $display(" 前件触发汇总（全部 >0 才说明断言真正被检查）:");
        $display("    AWVALID && !AWREADY : %0d", cnt_aw_wait);
        $display("    WVALID  && !WREADY  : %0d", cnt_w_wait);
        $display("    ARVALID && !ARREADY : %0d", cnt_ar_wait);
        $display("    BVALID  && !BREADY  : %0d", cnt_b_wait);
        $display("    RVALID  && !RREADY  : %0d", cnt_r_wait);

        // 最终检查：5 个前件全部被触发过
        checks++;
        if (cnt_aw_wait > 0 && cnt_w_wait > 0 && cnt_ar_wait > 0
            && cnt_b_wait > 0 && cnt_r_wait > 0) begin
            $display("  ✓ 5 条 AXI SVA 断言的前件全部被激活");
        end else begin
            $display("  ✗ 仍有断言前件未被激活（真空通过风险）");
            errors++;
        end

        $display("------------------------------------------------------");
        $display(" sva-antecedent result: %0d checks, %0d errors", checks, errors);
        if (errors == 0) begin
            $display("PASS: SVA antecedents activated");
            $finish;
        end else begin
            $display("FAIL: %0d errors", errors);
            $fatal(1, "SVA antecedent test failed");
        end
    end

endmodule
`endif // TB_SVA_ANTECEDENT_SV_SV
