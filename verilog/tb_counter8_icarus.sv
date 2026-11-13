// 新版 test bench：counter8 —— 本机 iverilog 可运行版
// 用 iverilog 12 支持的 SV 语法（logic / always_ff / enum / typedef），
// 断言用传统任务实现（iverilog 不支持 SVA 并发断言）。
// 完整 SV 版（interface + SVA + covergroup + class 随机）见 tb_counter8_sv.sv，
// 需 in irun/VCS 上运行。
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

    // ---- 断言任务（iverilog 替代 SVA） ----
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
        begin
            @(posedge clk);
            #1;
        end
    endtask

    // ---- 主流程 ----
    integer i;
    integer run_errors;
    initial begin
        $display("== SV(iverilog) test bench: counter8 ==");

        // 1. 复位
        wait_edge;  // 沿1：rst_n=0 -> count=0, carry=0
        check(1, 8'h00, count, 0);
        check(2, 1'b0, carry, 0);

        // 2. 使能关闭时保持
        rst_n <= 1; en <= 0;
        wait_edge; wait_edge;  // 沿2、3：保持 0
        check(3, 8'h00, count, 0);

        // 3. 计数 1..15（用枚举风格局部变量遍历段）
        en <= 1;
        for (i = 1; i <= 15; i++) begin
            wait_edge;  // 沿 i+3：推进
            check(4, i[7:0], count, i);
        end

        // 4. carry 行为：让计数到 255，验证回绕 + carry 高一拍
        $display("-- jumping to 255 --");
        // 继续计数到 254
        while (count != 8'hFE) begin
            wait_edge;
        end
        // 此时 count=FE(FE)，下一个沿：count+=1 -> FF，carry 下沿置位...
        // 实际 carry 在 count==FF 的下一拍为 1。我们走一步到 FF 再走一步看 carry。
        wait_edge;  // -> FF
        check(5, 8'hFF, count, 5);
        wait_edge;  // FF+1 -> 00, carry=1
        check(6, 8'h00, count, 6);
        check(7, 1'b1, carry, 7);
        wait_edge;  // 再一步：count=1, carry=0（因为 en=1 且 count!=FF）
        check(8, 8'h01, count, 8);
        check(9, 1'b0, carry, 9);

        // 5. 复位清零（非零值）
        en <= 0;            // 先停
        wait_edge;
        en <= 1;
        wait_edge;          // count: 01 -> 02
        rst_n <= 0;
        wait_edge;          // count -> 00
        check(10, 8'h00, count, 10);
        rst_n <= 1;

        run_errors = errors;
        if (run_errors == 0)
            $display("PASS: counter8 (iverilog) all %0d checks passed", 2 + 1 + 15 + 4 + 1 + 1 + 1 + 1);
        else
            $display("FAIL: %0d errors", run_errors);
        $finish;
    end

endmodule