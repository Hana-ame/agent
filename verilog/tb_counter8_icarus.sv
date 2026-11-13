// 新版 test bench：counter8 —— 本机 iverilog 12 可运行版（v2）
// 升级为「真·test bench」结构：
//   1) 确定性阶段：复位 / 保持 / 1..15 计数 / 回绕 / carry / 复位清零
//   2) 随机激励阶段：$random 驱动 en/rst，用「参考模型」逐周期自检
//      （参考模型 = 独立维护的期望 count/carry，与 DUT 输出逐拍比对）
// iverilog 不支持 SVA/covergroup/class，因此断言用传统 task 实现；
// 完整 SV 版见 tb_counter8_sv.sv（需 irun/VCS）。
`timescale 1ns / 1ps

module tb_counter8;

    localparam WIDTH = 8;

    // ---- 信号 ----
    logic        clk = 0;
    logic        rst_n = 0;
    logic        en = 0;
    logic [WIDTH-1:0] count;
    logic        carry;

    // ---- DUT ----
    counter8 #(
        .WIDTH(WIDTH)
    ) uut (
        .clk   (clk),
        .rst_n (rst_n),
        .en    (en),
        .count (count),
        .carry (carry)
    );

    // ---- 10ns 周期时钟 ----
    always #5 clk = ~clk;

    // ---- 检查 ----
    integer errors = 0;
    task check;
        input [31:0] tc;
        input [WIDTH-1:0] expected;
        input [WIDTH-1:0] actual;
        input [31:0] info;
        begin
            if (expected !== actual) begin
                $display("FAIL: tc=%0d expected=%h actual=%h (info=%0d) @%0t",
                         tc, expected, actual, info, $time);
                errors = errors + 1;
            end
        end
    endtask

    task wait_edge;
        @(posedge clk);
        #1;
    endtask

    // ---- 尝试运行 watchdog（iverilog 不支持 fork/join_any，故用可选模式） ----
    `ifdef TB_WATCHDOG
    initial begin
        // 若 TB 意外挂死，超时后以严重错误终止（可被 CI 捕获）
        // timescale 1ns/1ps：500_000_000 ps = 500 us，远大于正常仿真时长(约13us)
        #500_000_000;
        $display("TIMEOUT: test bench hung (>500 us)");
        $fatal(1, "watchdog timeout");
    end
    `endif

    // ---- 参考模型驱动 + 比对（随机阶段核心） ----
    // 时序约定与 DUT 一致：在 posedge 前的 count 上预测当前沿的更新结果，
    // 沿后 #1 采样实际值比对。
    // r: rst_n  e: en
    task drv_chk;
        input reg r;
        input reg e;
        input [31:0] tc;
        reg [WIDTH-1:0] exp_count;
        reg             exp_carry;
        begin
            // 先驱动（<= 在当前时刻末生效，下一拍沿前稳定）
            rst_n <= r;
            en    <= e;
            // 参考模型：用沿前 count 预测
            if (!r) begin
                exp_count = 8'h00;
                exp_carry = 1'b0;
            end else if (e) begin
                exp_count = count + 8'h01;
                exp_carry = (count == 8'hFF) ? 1'b1 : 1'b0;
            end else begin
                exp_count = count;
                exp_carry = 1'b0;
            end
            wait_edge;
            check(tc, exp_count, count, 0);
            check(tc, exp_carry, carry, 1);
        end
    endtask

    // ---- 上电复位序列（起始激励） ----
    // 形式：上电默认复位保持 → 释放 → 观察稳定态
    // 作用：保证 DUT 从确定状态开始，所有触发器归零后再进入测试
    task startup;
        begin
            $display("== STARTUP: power-on reset sequence ==");
            // 上电默认：复位拉低（低有效）、使能关闭
            rst_n = 1'b0;
            en    = 1'b0;
            repeat (2) wait_edge;        // 保持复位 2 周期，FF 全部归零
            check(0, 8'h00, count, 0);
            check(0, 1'b0,  carry, 1);
            // 释放复位，再观察 2 周期确保稳定
            $display("-- releasing reset --");
            rst_n = 1'b1;
            repeat (2) wait_edge;
            check(0, 8'h00, count, 0);
            check(0, 1'b0,  carry, 1);
        end
    endtask

    // ---- 收尾（结束激励形式） ----
    // 1) drain：把激励置回安全态，再跑几个周期冲洗边沿，确认最后状态
    // 2) finalize：按结果决定退出形式 —— 全过则 $finish（退出码 0），
    //    有错则 $fatal(1)（退出码 1，CI/脚本可捕获），并打印统计
    task drain;
        reg [WIDTH-1:0] hold;
        begin
            $display("== END OF STIMULUS: drain ==");
            // 激励回到默认：复位释放、使能关闭，跑 3 周期冲洗
            rst_n = 1'b1;
            en    = 1'b0;
            repeat (3) wait_edge;
            // 收尾断言：en=0 下 count 必须保持
            hold = count;              // 快照当前值
            wait_edge;                 // 再过一沿（en 仍为 0）
            check(11, hold, count, 0); // 应保持不动
        end
    endtask

    task finalize;
        begin
            $display("-- SUMMARY: %0d error(s) over %0d random cycles --",
                     errors, 1000);
            if (errors == 0) begin
                $display("PASS: counter8 (iverilog v2) all checks passed");
                $finish;                 // 退出码 0（iverilog 下 $finish(n) 恒 0）
            end else begin
                $display("FAIL: %0d errors", errors);
                $fatal(1, "test bench failed");   // 退出码 1，可被 CI 捕获
            end
        end
    endtask

    // ---- 主流程 ----
    integer i, j;
    reg rnd_r, rnd_e;
    initial begin
        $display("== SV(iverilog) test bench: counter8 (v2, random + reference model) ==");

        // ---- 起始激励：上电复位序列 ----
        startup;

        // ---- 阶段 1：确定性 1..15 ----
        en <= 1;
        for (i = 1; i <= 15; i = i + 1) begin
            wait_edge;                   // 沿 i+?（startup 后）
            check(4, i[7:0], count, i);
        end

        // ---- 阶段 2：确定性回绕 + carry ----
        while (count != 8'hFE) wait_edge;
        wait_edge;                       // -> FF
        check(5, 8'hFF, count, 5);
        wait_edge;                       // FF+1 -> 00, carry=1
        check(6, 8'h00, count, 6);
        check(7, 1'b1, carry, 7);
        wait_edge;                       // -> 01, carry=0
        check(8, 8'h01, count, 8);
        check(9, 1'b0, carry, 9);

        // ---- 阶段 3：随机激励 + 参考模型自检（1000 周期） ----
        $display("-- random phase: 1000 cycles with reference model --");
        for (j = 0; j < 1000; j = j + 1) begin
            rnd_r = (j % 97 == 0) ? 1'b0 : 1'b1;   // 每 97 拍来一次复位
            rnd_e = $random & 1;                   // 50% 使能
            drv_chk(rnd_r, rnd_e, 100 + j);
        end

        // ---- 阶段 4：确定性复位清零（非零值） ----
        en <= 0; wait_edge;              // 停
        en <= 1; wait_edge;              // count 推进一步
        if (count != 8'h00) begin
            rst_n <= 0;
            wait_edge;                   // -> 0
            check(10, 8'h00, count, 10);
            rst_n <= 1;
        end

        // ---- 结束激励：drain + finalize ----
        drain;
        finalize;
    end

endmodule