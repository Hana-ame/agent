// ============================================================================
// tb_assert_effectiveness_sv.sv：断言有效性验证（每条断言都必须能失败）
// ============================================================================
// 背景：
//   上轮发现 5 条 AXI 断言因前件从不成立而「真空通过」。
//   这提示需要逐条验证：每条断言在违规时是否真的会报错。
//
//   一个断言如果永远不会失败，它就没有保护作用 —— 无论它看起来多正确。
//
// 方法：
//   对每条 APB 断言，构造精确的违规时序，并检查：
//     - 违规注入 → 断言必须报错（用 $assertoff/on 隔离，避免 abort 整个仿真）
//     - 正常时序 → 断言不许误报
//
//   由于 Verilator 的 assert 失败会 $stop 终止仿真，本测试采用
//   「逐条独立子进程」不可行（testbench 内无法 fork 进程），
//   故改为：每条断言单独跑一次，通过 -DCASE=n 选择。
//
// 用法：仿真器参数 -DCASE=<1..5>
//   CASE=1  R1 违规（PENABLE without PSEL）
//   CASE=2  R3 违规（SETUP 阶段 PREADY）
//   CASE=3  R2 违规（ACCESS 提前退出）
//   CASE=4  R5 违规（PREADY 后未退出 ACCESS）
//   CASE=5  合法序列（应无断言失败）
// ============================================================================
`ifndef TB_ASSERT_EFFECTIVENESS_SV_SV
`define TB_ASSERT_EFFECTIVENESS_SV_SV
`timescale 1ns / 1ps

`ifndef CASE
  `define CASE 5
`endif

module tb_assert_effectiveness_sv #(
    parameter int AW = 8,
    parameter int DW = 32
);

    logic ACLK = 0, ARESETn = 0;
    always #5 ACLK = ~ACLK;

    apb_if #(.AW(AW), .DW(DW)) apb(ACLK);

    int errors = 0, checks = 0;

    task automatic do_reset();
        ARESETn = 0;
        apb.PSEL = 0; apb.PENABLE = 0; apb.PWRITE = 0;
        apb.PADDR = 0; apb.PWDATA = 0; apb.PRDATA = 0;
        apb.PREADY = 0; apb.PSLVERR = 0;
        repeat (4) @(posedge ACLK);
        ARESETn = 1;
        @(posedge ACLK);
    endtask

    task automatic tick();
        @(posedge ACLK); #1;
    endtask

    initial begin
        $display("======================================================");
        $display(" 断言有效性验证 CASE=%0d", `CASE);
        $display("======================================================");
        do_reset();

        case (`CASE)
            // ---------------------------------------------------------
            // CASE 1：违反 R1（PENABLE 无 PSEL）
            // 期望：ap_penable_requires_psel 报错
            // ---------------------------------------------------------
            1: begin
                $display("== 注入 R1 违规：PENABLE=1 但 PSEL=0 ==");
                tick();
                apb.PSEL = 0; apb.PENABLE = 1;
                tick();
                $display("FAIL: R1 断言未捕获违规（应报 PENABLE without PSEL）");
                $fatal(1);
            end

            // ---------------------------------------------------------
            // CASE 2：违反 R3（SETUP 阶段 PREADY）
            // 期望：ap_no_ready_in_setup 报错
            // ---------------------------------------------------------
            2: begin
                $display("== 注入 R3 违规：SETUP 阶段 PSEL=1,PENABLE=0,PREADY=1 ==");
                tick();
                apb.PSEL = 1; apb.PENABLE = 0; apb.PREADY = 1;
                tick();
                $display("FAIL: R3 断言未捕获违规（应报 PREADY during SETUP）");
                $fatal(1);
            end

            // ---------------------------------------------------------
            // CASE 3：违反 R2（ACCESS 未完成就退出）
            // 期望：ap_access_stable_until_ready 报错
            // ---------------------------------------------------------
            3: begin
                $display("== 注入 R2 违规：ACCESS 未完成就退出 ==");
                // 进入 ACCESS 且 PREADY=0
                tick(); apb.PSEL = 1; apb.PENABLE = 0; apb.PWRITE = 1;
                        apb.PADDR = 0; apb.PWDATA = 0;
                tick(); apb.PENABLE = 1; apb.PREADY = 0;
                tick();   // 此刻 PSEL&&PENABLE&&!PREADY 成立
                // 下一拍违规退出（无 PREADY）
                apb.PSEL = 0; apb.PENABLE = 0;
                tick();
                $display("FAIL: R2 断言未捕获违规（应报 ACCESS exited before PREADY）");
                $fatal(1);
            end

            // ---------------------------------------------------------
            // CASE 4：违反 R5（PREADY 后未退出 ACCESS）
            // 期望：ap_xfer_done_exits 报错
            // ---------------------------------------------------------
            4: begin
                $display("== 注入 R5 违规：PREADY=1 后仍保持 ACCESS ==");
                tick(); apb.PSEL = 1; apb.PENABLE = 0;
                tick(); apb.PENABLE = 1; apb.PREADY = 1;
                tick();   // 完成拍（PSEL&&PENABLE&&PREADY）
                // 违规：下一拍仍保持 PENABLE
                apb.PENABLE = 1; apb.PREADY = 0;
                tick();
                $display("FAIL: R5 断言未捕获违规（应报 PREADY high but ACCESS not exited）");
                $fatal(1);
            end

            // ---------------------------------------------------------
            // CASE 6：违反 R6（PSLVERR 在非完成拍拉高）
            // 期望：ap_slverr_only_on_complete 报错
            // ---------------------------------------------------------
            6: begin
                $display("== 注入 R6 违规：IDLE 阶段 PSLVERR=1 ==");
                tick();
                apb.PSEL = 0; apb.PENABLE = 0; apb.PSLVERR = 1;
                tick();
                $display("FAIL: R6 断言未捕获违规（应报 PSLVERR outside completion）");
                $fatal(1);
            end

            // ---------------------------------------------------------
            // CASE 5：合法序列（所有断言都不应误报）
            // ---------------------------------------------------------
            default: begin
                $display("== 合法 APB 序列（断言不应误报）==");
                // SETUP
                tick(); apb.PSEL = 1; apb.PENABLE = 0; apb.PWRITE = 1;
                        apb.PADDR = 8'h04; apb.PWDATA = 32'hDEAD_BEEF;
                // ACCESS（插 2 拍 wait）
                tick(); apb.PENABLE = 1; apb.PREADY = 0;
                tick(); apb.PREADY = 0;
                // 完成
                tick(); apb.PREADY = 1;
                tick(); apb.PSEL = 0; apb.PENABLE = 0; apb.PREADY = 0;
                tick(); tick();

                checks++;
                $display("  ✓ 合法序列完成，无断言失败");
                $display("PASS: legal sequence (no assertion violation)");
                $finish;
            end
        endcase
    end

endmodule
`endif // TB_ASSERT_EFFECTIVENESS_SV_SV
