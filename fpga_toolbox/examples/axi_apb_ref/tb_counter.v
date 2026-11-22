// counter 的 test bench（修正版）
// 覆盖：复位、使能保持、逐周期计数、回绕（15->0）、同步复位清零
// 时序约定：posedge 后的 #1 采样到的是「本次沿更新后的值」，
// 因此每个阶段的期望值按紧邻前一个沿的更新结果来写。
`timescale 1ns / 1ps

module tb_counter;

    reg        clk = 0;
    reg        rst_n = 0;
    reg        en = 0;
    wire [3:0] count;

    counter #(
        .WIDTH(4)
    ) uut (
        .clk   (clk),
        .rst_n (rst_n),
        .en    (en),
        .count (count)
    );

    // ---- 10ns 周期时钟 ----
    always #5 clk = ~clk;

    // ---- 任务：等一个时钟沿 ----
    task wait_edge;
        begin
            @(posedge clk);
            #1; // 越过采样沿，读稳定值
        end
    endtask

    // ---- 检查（msg 用可综合的常量做匹配，避免 task 字符串传参问题）----
    integer errors = 0;
    task check;
        input [15:0] expected;
        input [3:0]  actual;
        input [3:0]  step;
        input integer phase;
        begin
            if (expected !== actual) begin
                $display("FAIL: phase=%0d step=%0d expected=%b actual=%b @%0t",
                         phase, step, expected, actual, $time);
                errors = errors + 1;
            end
        end
    endtask

    // ---- 主流程 ----
    integer i;
    integer run_errors;
    initial begin
        $dumpfile("tb_counter.vcd");
        $dumpvars(0, tb_counter);

        $display("== 1. reset: count should be 0 ==");
        wait_edge; // 沿1：rst_n=0 -> count=0
        check(4'b0000, count, 0, 1);

        // 进入计数前，先让一个沿把 rst_n=1 生效
        rst_n = 1; en = 0;
        wait_edge; // 沿2：rst 释放，en=0 -> count 保持 0

        $display("== 2. en=0 holds value ==");
        wait_edge; // 沿3：en=0 -> 保持 0
        check(4'b0000, count, 0, 2);

        $display("== 3. count 1..15, wrap to 0 while en=1 ==");
        en = 1;
        // en 在沿 3 之后置位，从沿 4 起每个沿推进一步：
        // 第 i 次采样（沿 i+4）得到 (i+1) mod 16 → 1,2,...,15,0（回绕在 i=15 验证）
        for (i = 0; i < 16; i = i + 1) begin
            wait_edge;
            check((i + 1) % 16, count, i + 1, 3);
        end

        $display("== 4. synchronous reset clears a non-zero count while en=1 ==");
        // 此刻 count=0（沿19 回绕）；再推一步到 1，然后断言复位看它被清零
        wait_edge;          // 沿20: 0 -> 1
        check(4'b0001, count, 0, 4); // 先确认停在非零值
        rst_n = 0;
        wait_edge;          // 沿21: reset 生效 -> 0
        check(4'b0000, count, 0, 4);
        rst_n = 1;

        // ---- 汇总（1 复位 + 1 保持 + 16 计数/回绕 + 2 复位清零 = 20 项） ----
        run_errors = errors;
        if (run_errors == 0)
            $display("PASS: all 20 checks passed");
        else
            $display("FAIL: %0d checks failed", run_errors);
        $finish;
    end

endmodule