// 完整 SystemVerilog test bench：counter8
// 现代验证方法学写法 —— 需 irun / VCS 等支持完整 SV 的工具运行。
// 本机 iverilog 12 编译不过（不支持 clocking / SVA / covergroup / class 随机）。
//
// 包含：
//   interface + clocking (可在 SV 环境中同步驱动)
//   SVA 并发断言（assert property）
//   covergroup 功能覆盖率
//   class + 随机约束激励（rand / constraint / dist）
`timescale 1ns / 1ps

// ---- interface：封装 DUT 端口 ----
interface counter_if #(parameter WIDTH = 8) (input logic clk);
    logic        rst_n;
    logic        en;
    logic [WIDTH-1:0] count;
    logic        carry;

    clocking cb @(posedge clk);
        default input #1step output #1;
        output rst_n, en;
        input  count, carry;
    endclocking

    // ---- SVA 并发断言 ----
    // 复位后 count==0
    ap_reset: assert property (@(posedge clk) !rst_n |=> count == 0);
    // 非满值时使能则递增
    ap_incr: assert property (@(posedge clk) rst_n && en && (count != '1) |=> count == $past(count) + 1);
    // 满值时使能则回绕到 0
    ap_wrap: assert property (@(posedge clk) rst_n && en && (count == '1) |=> count == 0);
    // 使能关闭保持
    ap_hold: assert property (@(posedge clk) rst_n && !en |=> count == $past(count));
    // 回绕时 carry 高一拍
    ap_carry: assert property (@(posedge clk) rst_n && (count == '1) |=> carry == 1);
endinterface

// ---- class + 随机约束：80% 使能 ----
class Stimulus;
    rand bit en;
    constraint en_prob { en dist {1 := 80, 0 := 20}; }
endclass

// ---- test bench ----
module tb_counter8;
    localparam WIDTH = 8;
    logic clk = 0;

    counter_if #(WIDTH) uif (clk);
    counter8    #(WIDTH) dut (
        .clk   (uif.clk),
        .rst_n (uif.rst_n),
        .en    (uif.en),
        .count (uif.count),
        .carry (uif.carry)
    );

    // ---- 功能覆盖率 ----
    covergroup cg @(posedge clk);
        cp_rst: coverpoint uif.rst_n;
        cp_en:  coverpoint uif.en;
        cp_count: coverpoint uif.count {
            bins low  = {[0:31]};
            bins mid  = {[32:223]};
            bins high = {[224:255]};
            bins wrap = {0};
        }
        cross cp_en, cp_count;
    endgroup
    cg cg_inst = new();

    // ---- 时钟 ----
    always #5 clk = ~clk;

    // ---- 主流程：复位 + 随机激励 + 显式回绕/复位检查 ----
    int cycles;
    int errors = 0;
    Stimulus stim = new();

    initial begin
        $display("== SV test bench: counter8 (SVA + coverage + random) ==");

        // 复位
        uif.rst_n <= 1'b0; uif.en <= 1'b0;
        repeat (2) @(posedge clk);
        uif.rst_n <= 1'b1;

        // 512 周期随机激励
        for (cycles = 0; cycles < 512; cycles++) begin
            if (!stim.randomize())
                $error("randomize failed");
            uif.en <= stim.en;
            @(posedge clk);
            cg_inst.sample();
        end

        // 显式回绕检查
        uif.en <= 1'b1;
        for (cycles = 0; cycles < 260; cycles++) begin
            if (uif.count == 8'hFF) begin
                @(posedge clk);
                if (uif.count !== 8'h00 || uif.carry !== 1'b1) begin
                    $display("WRAP FAIL: count=%h carry=%b", uif.count, uif.carry);
                    errors++;
                end
                break;
            end
            @(posedge clk);
        end

        // 复位清零检查
        uif.en <= 1'b1;
        @(posedge clk);
        if (uif.count == 8'h02) begin
            uif.rst_n <= 1'b0;
            @(posedge clk);
            if (uif.count !== 8'h00) begin
                $display("RST FAIL");
                errors++;
            end
            uif.rst_n <= 1'b1;
        end

        if (errors == 0)
            $display("PASS: counter8 SV test bench, %0d cycles random, SVA/coverage active", cycles);
        else
            $display("FAIL: %0d errors", errors);
        $finish;
    end
endmodule