// 完整 SystemVerilog test bench：counter8（v2，标准现代验证结构）
// 需 irun / VCS 等完整 SV 工具运行；本机 iverilog 12 编译不过（无 SVA/covergroup/class）。
//
// 结构：
//   1) interface + clocking: 汇总 DUT 端口 + 同步驱动通道
//   2) class 随机激励: rand en / rand rst，constraint 控制概率
//   3) 参考模型: scoreboard 用沿前 count 预测下一拍 count/carry，与 DUT 比对
//   4) SVA: 5 条并发断言（复位/递增/回绕/保持/carry）
//   5) covergroup: 功能覆盖率（含 cross）
//   6) 显式回绕 + 复位清零检查
`timescale 1ns / 1ps

// ---- interface：封装 DUT 端口 + 断言 + 采样 ----
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
    ap_reset:  assert property (@(posedge clk) !rst_n |=> count == 0);
    ap_incr:   assert property (@(posedge clk) rst_n && en && (count != '1) |=> count == $past(count) + 1);
    ap_wrap:   assert property (@(posedge clk) rst_n && en && (count == '1) |=> count == 0);
    ap_hold:   assert property (@(posedge clk) rst_n && !en |=> count == $past(count));
    ap_carry:  assert property (@(posedge clk) rst_n && (count == '1) |=> carry == 1);
endinterface

// ---- 随机激励类 ----
class Stimulus;
    rand bit rst;   // 低有效复位（1=释放）
    rand bit en;
    constraint rst_prob { rst dist {1 := 97, 0 := 3}; }  // 3% 周期复位
    constraint en_prob  { en  dist {1 := 70, 0 := 30}; } // 70% 使能
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

    // ---- 覆盖率 ----
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

    // ---- 参考模型：预测下一拍 ----
    function void predict(input logic r, input logic e,
                          input logic [WIDTH-1:0] cur,
                          output logic [WIDTH-1:0] nx,
                          output logic nx_carry);
        if (!r) begin
            nx = 0; nx_carry = 0;
        end else if (e) begin
            nx = cur + 1'b1;
            nx_carry = (cur == '1);
        end else begin
            nx = cur; nx_carry = 0;
        end
    endfunction

    // ---- 统计 ----
    int cycles = 0;
    int errors = 0;

    // ---- 起始激励：上电复位序列 (SV 风格，经 clocking 驱动) ----
    task startup();
        $display("== STARTUP: power-on reset sequence ==");
        uif.cb.rst_n <= 1'b0;   // 上电默认复位
        uif.cb.en    <= 1'b0;
        repeat (2) @(posedge clk);
        // SVA 断言 ap_reset 会验证复位后 count==0；此处显式确认
        if (uif.count !== 0) begin $display("STARTUP FAIL"); errors++; end
        $display("-- releasing reset --");
        uif.cb.rst_n <= 1'b1;
        repeat (2) @(posedge clk);
        if (uif.count !== 0) begin $display("STARTUP FAIL"); errors++; end
    endtask

    // ---- 结束激励 1：drain（收尾冲洗 + 最后保持断言） ----
    task drain();
        logic [WIDTH-1:0] hold;
        $display("== END OF STIMULUS: drain ==");
        uif.cb.rst_n <= 1'b1;
        uif.cb.en    <= 1'b0;
        repeat (3) @(posedge clk);
        hold = uif.count;
        @(posedge clk);                    // en=0 下应保持
        if (uif.count !== hold) begin
            $display("DRAIN FAIL: count changed %h -> %h", hold, uif.count);
            errors++;
        end
    endtask

    // ---- 结束激励 2：finalize（统计 + 退出形式） ----
    //   通过 -> $finish（退出码 0）
    //   失败 -> $fatal(1)（退出码 1，CI/脚本可捕获）
    task finalize();
        $display("-- SUMMARY: %0d error(s) over %0d random cycles --", errors, cycles);
        if (errors == 0) begin
            $display("PASS: counter8 SV, all checks passed");
            $finish;
        end else begin
            $display("FAIL: %0d errors", errors);
            $fatal(1, "test bench failed");
        end
    endtask

    initial begin
        $display("== SV test bench: counter8 v2 (SVA + coverage + reference model) ==");

        // ---- 起始激励：上电复位序列 ----
        startup();

        // 随机激励循环（2000 周期），每拍驱动 + 参考模型比对
        Stimulus stim = new();
        logic [WIDTH-1:0] nxt; logic nxt_carry;
        for (cycles = 0; cycles < 2000; cycles++) begin
            if (!stim.randomize())
                $error("randomize failed at %0d", cycles);
            uif.rst_n <= stim.rst;
            uif.en    <= stim.en;
            // 参考模型预测（基于沿前值）
            predict(stim.rst, stim.en, uif.count, nxt, nxt_carry);
            @(posedge clk);
            if (uif.count !== nxt) begin
                $display("SCOREBOARD FAIL: cycle=%0d exp count=%h got=%h", cycles, nxt, uif.count);
                errors++;
            end
            if (uif.carry !== nxt_carry) begin
                $display("SCOREBOARD FAIL: cycle=%0d exp carry=%b got=%b", cycles, nxt_carry, uif.carry);
                errors++;
            end
            cg_inst.sample();
        end

        // 显式回绕检查
        $display("-- deterministic wrap check --");
        uif.en <= 1'b1; uif.rst_n <= 1'b1;
        while (uif.count != 8'hFF) @(posedge clk);
        @(posedge clk);
        if (uif.count !== 8'h00 || uif.carry !== 1'b1) begin
            $display("WRAP FAIL: count=%h carry=%b", uif.count, uif.carry);
            errors++;
        end

        // 显式复位清零检查
        $display("-- deterministic reset check --");
        uif.en <= 1'b1;
        @(posedge clk);
        uif.rst_n <= 1'b0;
        @(posedge clk);
        if (uif.count !== 8'h00) begin
            $display("RST FAIL: count=%h", uif.count);
            errors++;
        end
        uif.rst_n <= 1'b1;

        $display("Coverage: rst=%.1f%% en=%.1f%%", cg_inst.cp_rst.get_coverage(), cg_inst.cp_en.get_coverage());
        $display("Coverage count bins: %.1f%%", cg_inst.cp_count.get_coverage());

        // ---- 结束激励：drain + finalize ----
        drain();
        finalize();
    end
endmodule