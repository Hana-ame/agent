// ============================================================================
// tb_monitor_rules.sv：监视器规则逐条验证（每条规则都要「能抓到」）
// ============================================================================
// 上轮只验证了 P1 / P3 两条规则，其余 9 条从未被单独证明有效。
// 本测试对每条规则构造精确的信号时序，逐一验证：
//   - 违规注入 → 必须报出对应规则
//   - 不注入   → 不许误报
//
// 方法：直接驱动接口信号（不用 DUT），因为要精确控制每一拍。
//       SVA 断言用 $assertoff 关闭，隔离验证监视器本身。
//
// 规则清单
//   A1.AWVALID / A1.WVALID / A1.ARVALID   VALID 未握手就撤销
//   A2.AWADDR  / A2.WDATA  / A2.ARADDR    等待期间地址/数据变化
//   P1.PENABLE                            PENABLE 无 PSEL
//   P2.PHASE                              ACCESS 提前退出
//   P3.SETUP                              SETUP 阶段 PREADY
//   P4.PADDR   / P4.PWRITE                等待期间地址/控制变化
// ============================================================================
`ifndef TB_MONITOR_RULES_SV
`define TB_MONITOR_RULES_SV
`timescale 1ns / 1ps

module tb_monitor_rules #(
    parameter int AW = 8,
    parameter int DW = 32
);

    logic ACLK = 0, ARESETn = 0;
    always #5 ACLK = ~ACLK;

    axi_lite_if #(.AW(AW), .DW(DW)) axi(ACLK);
    apb_if      #(.AW(AW), .DW(DW)) apb(ACLK);

    axi_apb_monitor #(.AW(AW), .DW(DW), .NAME("RULES"), .VERBOSE(0)) mon (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
    );

    int checks = 0;
    int errors = 0;

    // ---- 复位到干净状态 ----
    task automatic idle_bus();
        axi.AWVALID = 0; axi.WVALID = 0; axi.ARVALID = 0;
        axi.BREADY = 0;  axi.RREADY = 0;
        axi.AWADDR = 0;  axi.WDATA = 0;   axi.ARADDR = 0;
        apb.PSEL = 0; apb.PENABLE = 0; apb.PWRITE = 0;
        apb.PADDR = 0; apb.PWDATA = 0; apb.PRDATA = 0;
        apb.PREADY = 0; apb.PSLVERR = 0;
    endtask

    task automatic do_reset();
        mon.rule_clear();
        ARESETn = 0;
        idle_bus();
        repeat (3) @(posedge ACLK);
        ARESETn = 1;
        @(posedge ACLK);
        idle_bus();
        @(posedge ACLK);
    endtask

    // 拍一次时钟并让监视器采样
    task automatic tick();
        @(posedge ACLK); #1;
    endtask

    // ---- 断言辅助：本段应新增 N 处违规 ----
    int err_base;

    // 期望「tag 这条规则」被触发 delta 次。
    // 关键：必须校验规则名，不能只看违规总数 ——
    // 一次注入可能同时触发多条规则（如 ACCESS 提前退出会同时
    // 触发 P1/P2/P3），数量对了不代表目标规则生效。
    task automatic expect_violation(input string tag, input int delta = 1);
        checks++;
        begin
            int cnt;
            cnt = mon.rule_hits.exists(tag) ? mon.rule_hits[tag] : 0;
            if (mon.errors - err_base >= 1 && cnt == delta) begin
                $display("  ✓ %-12s 规则命中 %0d 次（预期 %0d，总违规 %0d）",
                         tag, cnt, delta, mon.errors - err_base);
            end else begin
                $display("  ✗ %-12s 规则命中 %0d 次（预期 %0d，总违规 %0d）—— 目标规则未生效",
                         tag, cnt, delta, mon.errors - err_base);
                errors++;
            end
        end
        err_base = mon.errors;
    endtask

    task automatic expect_clean(input string tag);
        checks++;
        if (mon.errors == err_base) begin
            $display("  ✓ %-12s 无违规（正确）", tag);
        end else begin
            $display("  ✗ %-12s 误报 %0d 处违规", tag, mon.errors - err_base);
            errors++;
        end
        err_base = mon.errors;
    endtask

    // ==================================================================
    // A1：VALID 未握手就撤销
    // ==================================================================
    task automatic test_a1_awvalid();
        $display("== A1.AWVALID：AWVALID 未等 AWREADY 就撤销 ==");
        do_reset(); err_base = mon.errors;

        tick(); axi.AWVALID = 1; axi.AWADDR = 8'h10;
        tick();                      // 监视器看到 AWVALID=1
        axi.AWADDR = 8'h10;          // 保持地址
        tick(); axi.AWVALID = 0;     // 撤销（AWREADY 一直是 0）
        tick();
        expect_violation("A1.AWVALID");
    endtask

    task automatic test_a1_wvalid();
        $display("== A1.WVALID：WVALID 未等 WREADY 就撤销 ==");
        do_reset(); err_base = mon.errors;

        tick(); axi.WVALID = 1; axi.WDATA = 32'hAAAA_0001;
        tick();
        tick(); axi.WVALID = 0;
        tick();
        expect_violation("A1.WVALID");
    endtask

    task automatic test_a1_arvalid();
        $display("== A1.ARVALID：ARVALID 未等 ARREADY 就撤销 ==");
        do_reset(); err_base = mon.errors;

        tick(); axi.ARVALID = 1; axi.ARADDR = 8'h20;
        tick();
        tick(); axi.ARVALID = 0;
        tick();
        expect_violation("A1.ARVALID");
    endtask

    // ==================================================================
    // A2：等待期间地址/数据变化
    // ==================================================================
    task automatic test_a2_awaddr();
        $display("== A2.AWADDR：AWVALID 保持期间改地址 ==");
        do_reset(); err_base = mon.errors;

        tick(); axi.AWVALID = 1; axi.AWADDR = 8'h10;
        tick();
        axi.AWADDR = 8'h20;          // 违规：地址变了
        tick();
        axi.AWVALID = 0;
        tick();
        // A1 也会触发（未握手撤销），故预期 ≥1
        checks++;
        if (mon.errors - err_base >= 1 && grep_violation("A2")) begin
            $display("  ✓ A2.AWADDR  捕获地址变化违规");
        end else begin
            $display("  ✗ A2.AWADDR  未捕获地址变化违规");
            errors++;
        end
        err_base = mon.errors;
    endtask

    // A2.ARADDR：ARVALID 保持期间改地址（此前只测了 AWADDR/WDATA，
    // 导致监视器里这条规则从未被执行，是真实的测试缺口）
    task automatic test_a2_araddr();
        $display("== A2.ARADDR：ARVALID 保持期间改地址 ==");
        do_reset(); err_base = mon.errors;

        tick(); axi.ARVALID = 1; axi.ARADDR = 8'h10;
        tick();
        axi.ARADDR = 8'h20;          // 违规：地址变了
        tick();
        axi.ARVALID = 0;
        tick();
        checks++;
        if (mon.errors - err_base >= 1 && grep_violation("A2")) begin
            $display("  ✓ A2.ARADDR  捕获地址变化违规");
        end else begin
            $display("  ✗ A2.ARADDR  未捕获地址变化违规");
            errors++;
        end
        err_base = mon.errors;
    endtask

    task automatic test_a2_wdata();
        $display("== A2.WDATA：WVALID 保持期间改数据 ==");
        do_reset(); err_base = mon.errors;

        tick(); axi.WVALID = 1; axi.WDATA = 32'h1111_1111;
        tick();
        axi.WDATA = 32'h2222_2222;   // 违规
        tick();
        axi.WVALID = 0;
        tick();
        checks++;
        // 精确校验：必须是 A2.WDATA 这条规则被触发，
        // 而不是任意其他规则（A1 未握手撤销也会计数）
        if (mon.errors - err_base >= 1 && grep_violation("A2.WDATA")) begin
            $display("  ✓ A2.WDATA   捕获数据变化违规");
        end else begin
            $display("  ✗ A2.WDATA   未捕获数据变化违规");
            errors++;
        end
        err_base = mon.errors;
    endtask

    // ==================================================================
    // P1：PENABLE 无 PSEL
    // ==================================================================
    task automatic test_p1();
        $display("== P1.PENABLE：PENABLE=1 但 PSEL=0 ==");
        do_reset(); err_base = mon.errors;

        tick(); apb.PSEL = 0; apb.PENABLE = 1;
        tick();
        apb.PENABLE = 0;
        tick();
        expect_violation("P1.PENABLE");
    endtask

    // ==================================================================
    // P2：ACCESS 提前退出
    // ==================================================================
    task automatic test_p2();
        $display("== P2.PHASE：ACCESS 阶段未等 PREADY 就退出 ==");
        do_reset(); err_base = mon.errors;

        // 进入 ACCESS 但 PREADY=0（等待）
        tick(); apb.PSEL = 1; apb.PENABLE = 0; apb.PWRITE = 1;
                apb.PADDR = 8'h00; apb.PWDATA = 32'h1234;
        tick(); apb.PENABLE = 1; apb.PREADY = 0;
        tick();                       // 监视器记录 prev_access=1, prev_ready=0
        // 违规：ACCESS 未完成就退出（PSEL/PENABLE 同时掉）
        apb.PSEL = 0; apb.PENABLE = 0; apb.PREADY = 0;
        tick();
        expect_violation("P2.PHASE");
    endtask

    // ==================================================================
    // P3：SETUP 阶段 PREADY
    // ==================================================================
    task automatic test_p3();
        $display("== P3.SETUP：SETUP 阶段就回 PREADY ==");
        do_reset(); err_base = mon.errors;

        tick(); apb.PSEL = 1; apb.PENABLE = 0; apb.PREADY = 1;
        tick();
        apb.PSEL = 0; apb.PREADY = 0;
        tick();
        expect_violation("P3.SETUP");
    endtask

    // ==================================================================
    // P4：等待期间 PADDR / PWRITE 变化
    // ==================================================================
    task automatic test_p4_paddr();
        $display("== P4.PADDR：ACCESS 等待期间改地址 ==");
        do_reset(); err_base = mon.errors;

        tick(); apb.PSEL = 1; apb.PENABLE = 0; apb.PWRITE = 1;
                apb.PADDR = 8'h00; apb.PWDATA = 32'h1111;
        tick(); apb.PENABLE = 1; apb.PREADY = 0;
        tick();                        // prev_access=1, prev_ready=0
        apb.PADDR = 8'h40;             // 违规：等待期间地址变
        tick();
        apb.PSEL = 0; apb.PENABLE = 0; apb.PADDR = 0;
        tick();
        checks++;
        if (mon.errors - err_base >= 1 && grep_violation("P4.PADDR")) begin
            $display("  ✓ P4.PADDR   捕获地址变化违规");
        end else begin
            $display("  ✗ P4.PADDR   未捕获地址变化违规");
            errors++;
        end
        err_base = mon.errors;
    endtask

    task automatic test_p4_pwrite();
        $display("== P4.PWRITE：ACCESS 等待期间改读写方向 ==");
        do_reset(); err_base = mon.errors;

        tick(); apb.PSEL = 1; apb.PENABLE = 0; apb.PWRITE = 1;
                apb.PADDR = 8'h00; apb.PWDATA = 32'h1111;
        tick(); apb.PENABLE = 1; apb.PREADY = 0;
        tick();
        apb.PWRITE = 0;                // 违规：等待期间改方向
        tick();
        apb.PSEL = 0; apb.PENABLE = 0;
        tick();
        checks++;
        if (mon.errors - err_base >= 1 && grep_violation("P4.PWRITE")) begin
            $display("  ✓ P4.PWRITE  捕获方向变化违规");
        end else begin
            $display("  ✗ P4.PWRITE  未捕获方向变化违规");
            errors++;
        end
        err_base = mon.errors;
    endtask

    // ==================================================================
    // 合法操作：全部规则都不应误报
    // ==================================================================
    // ==================================================================
    // A3：串行桥不允许 AW 与 AR 同时请求
    // ==================================================================
    task automatic test_a3_channel();
        $display("== A3.CHANNEL：AWVALID 与 ARVALID 同时拉高 ==");
        do_reset(); err_base = mon.errors;

        tick(); axi.AWVALID = 1; axi.AWADDR = 8'h10;
                axi.ARVALID = 1; axi.ARADDR = 8'h20;
        tick();
        tick();
        axi.AWVALID = 0; axi.ARVALID = 0;
        tick();
        checks++;
        if (mon.errors - err_base >= 1 && grep_violation("A3.CHANNEL")) begin
            $display("  ✓ A3.CHANNEL  捕获读写同时请求");
        end else begin
            $display("  ✗ A3.CHANNEL  未捕获读写同时请求");
            errors++;
        end
        err_base = mon.errors;
    endtask

    task automatic test_legal_all();
        $display("== 合法事务：所有规则均不应误报 ==");
        do_reset(); err_base = mon.errors;

        // ---- 合法 AXI 写：AW+W 同拍发起，同拍 READY（真握手）----
        // 注意：AWREADY/WREADY 是 slave 输出，本测试作为环境需要驱动它们
        tick();
        axi.AWVALID = 1; axi.AWADDR = 8'h00;
        axi.WVALID  = 1; axi.WDATA  = 32'hCAFE_0001;
        axi.AWREADY = 1; axi.WREADY = 1;    // 同拍握手完成
        tick();
        // 握手已完成（上一拍 valid&&ready），本拍撤销 VALID 合法
        axi.AWVALID = 0; axi.WVALID = 0;
        axi.AWREADY = 0; axi.WREADY = 0;
        tick();

        // ---- 合法 APB 写：SETUP → ACCESS → PREADY ----
        tick(); apb.PSEL = 1; apb.PENABLE = 0; apb.PWRITE = 1;
                apb.PADDR = 8'h00; apb.PWDATA = 32'hCAFE_0001;
        tick(); apb.PENABLE = 1; apb.PREADY = 0;
        tick(); apb.PREADY = 1;              // ACCESS 完成
        tick(); apb.PSEL = 0; apb.PENABLE = 0; apb.PREADY = 0;
        tick();

        // ---- 合法 AXI 读：AR+READY 同拍握手 ----
        tick();
        axi.ARVALID = 1; axi.ARADDR = 8'h00;
        axi.ARREADY = 1; axi.RREADY = 1;
        tick();
        axi.ARVALID = 0; axi.ARREADY = 0; axi.RREADY = 0;
        tick();

        expect_clean("全部规则");
    endtask

    // 精确查询某规则是否被触发过
    // 此前是空实现（永远返回 1），导致「规则失效也测不出来」——
    // 这是真实盲区：M28 变异（A2.AWADDR 规则删掉）本应被发现却存活。
    // 现在改为调用监视器的 rule_hit() 做精确匹配。
    function automatic bit grep_violation(input string kw);
        return mon.rule_hit(kw);
    endfunction

    // ==================================================================
    initial begin
        $display("======================================================");
        $display(" monitor rules: 逐条验证（注入 + 无误报）");
        $display("======================================================");

        idle_bus();
        do_reset();
        $assertoff(0, tb_monitor_rules);   // 关闭 SVA，隔离验证监视器

        test_a1_awvalid();
        test_a1_wvalid();
        test_a1_arvalid();
        test_a2_awaddr();
        test_a2_araddr();
        test_a2_wdata();
        test_p1();
        test_p2();
        test_p3();
        test_p4_paddr();
        test_p4_pwrite();
        test_a3_channel();
        test_legal_all();

        $display("------------------------------------------------------");
        $display(" rules result: %0d checks, %0d errors, %0d violations detected",
                 checks, errors, mon.errors);
        if (errors == 0) begin
            $display("PASS: all monitor rules verified");
            $finish;
        end else begin
            $display("FAIL: %0d rule checks failed", errors);
            $fatal(1, "monitor rules test failed");
        end
    end

endmodule
`endif // TB_MONITOR_RULES_SV
