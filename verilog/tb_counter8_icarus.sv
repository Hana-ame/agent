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

    // ---- 主流程 ----
    integer i, j;
    reg rnd_r, rnd_e;
    integer run_errors;
    initial begin
        $display("== SV(iverilog) test bench: counter8 (v2, random + reference model) ==");

        // ---- 阶段 0：复位 ----
        wait_edge;                       // 沿1：rst_n=0 -> 0
        check(1, 8'h00, count, 0);
        check(2, 1'b0, carry, 0);

        // ---- 阶段 1：确定性 1..15 ----
        rst_n <= 1; en <= 0;
        wait_edge; wait_edge;            // 沿2、3：释放复位保持
        check(3, 8'h00, count, 0);
        en <= 1;
        for (i = 1; i <= 15; i = i + 1) begin
            wait_edge;                   // 沿 i+3
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

        run_errors = errors;
        if (run_errors == 0)
            $display("PASS: counter8 (iverilog v2) all checks passed");
        else
            $display("FAIL: %0d errors", run_errors);
        $finish;
    end

endmodule