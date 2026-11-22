// ============================================================================
// tb_dual_legal_sv.sv：并发桥的「合法并发」验证
// ============================================================================
// 关键概念：AXI 规定**每个通道同一时刻只有 1 笔事务**。
//   所以「两个写同时」「两个读同时」在 AXI 上是非法的，
//   用 fork 并发调用同一个 host 会破坏握手时序（不是 DUT 缺陷）。
//
// 并发桥真正的并发是：
//   写通道（AW/W/B）与读通道（AR/R）**同时进行**，
//   由桥内两个独立 FSM（W-FSM / R-FSM）并行处理。
//
// 本测试验证 3 项：
//   [1] 写 0x20 + 同时读 0x10：两者都正确（不同寄存器，无数据依赖）
//   [2] 验证 [1] 的写真的落盘（不是假成功）
//   [3] 连续 5 轮 写+读 并发，验证无累积错误
//
// 反面记录（曾误判）：
//   最初写成 fork 并发两个 host.execute()，观测到数据串扰，
//   一度以为是并发桥缺陷。核查后确认是测试方法错误——
//   同一 host 被并发调用破坏了 AXI 单通道单事务约束。
// ============================================================================
`timescale 1ns/1ps
module tb_dual_legal_sv;
  localparam int AW=8,DW=32,NREG=16;
  logic ACLK=0, ARESETn=0;
  always #5 ACLK=~ACLK;
  axi_lite_if #(AW,DW) axi(ACLK);
  apb_if #(AW,DW) apb(ACLK);
  axi_apb_bridge_dual_sv #(AW,DW) bridge(.ACLK(ACLK),.ARESETn(ARESETn),.axi(axi),.apb(apb));
  apb_slave_regs_sv #(AW,DW,NREG,1) slave(.PCLK(ACLK),.PRESETn(ARESETn),.apb(apb));
  axi_txn_host #(AW,DW) host(.ACLK(ACLK),.ARESETn(ARESETn),.axi(axi));

  import bus_txn_pkg::*; import axi_defs_pkg::*;
  int errors=0, checks=0;

  initial begin
    txn_t tw, tr; rsp_t rw, rr;
    $display("=== 并发桥：合法并发（写通道 + 读通道）===");
    axi.AWVALID=0; axi.WVALID=0; axi.ARVALID=0; axi.BREADY=0; axi.RREADY=0;
    ARESETn=0; repeat(5) @(posedge ACLK); ARESETn=1; @(posedge ACLK);

    // 预置
    tw='0; tw.write=1; tw.addr=8'h10; tw.data=32'h1111_1111;
    host.execute(tw, rw);
    tw='0; tw.write=1; tw.addr=8'h14; tw.data=32'h2222_2222;
    host.execute(tw, rw);
    tw='0; tw.write=1; tw.addr=8'h18; tw.data=32'h3333_3333;
    host.execute(tw, rw);

    // [1] 写 0x20 + 同时读 0x10（不同寄存器，无依赖）
    $display("== [1] 写0x20 + 读0x10 同时 ==");
    tw='0; tw.write=1; tw.addr=8'h20; tw.data=32'hABCD_1234;
    tr='0; tr.write=0; tr.addr=8'h10; tr.data='0;
    fork
      begin host.execute(tw, rw); end
      begin host.execute(tr, rr); end
    join
    checks++;
    if (rw.resp===RESP_OKAY && rr.resp===RESP_OKAY && rr.data===32'h1111_1111)
      $display("  ✓ 写+读同时：写OKAY，读数据正确（%h）", rr.data);
    else begin
      $display("  ✗ 失败：写resp=%0d 读resp=%0d 读data=%h", rw.resp, rr.resp, rr.data);
      errors++;
    end

    // [2] 验证 [1] 的写真的生效
    tr='0; tr.write=0; tr.addr=8'h20; tr.data='0;
    host.execute(tr, rr);
    checks++;
    if (rr.resp===RESP_OKAY && rr.data===32'hABCD_1234)
      $display("  ✓ 并发写的数据正确落盘（%h）", rr.data);
    else begin $display("  ✗ 数据丢失：%h", rr.data); errors++; end

    // [3] 写0x24 + 读0x14（连续多轮，验证无累积错误）
    $display("== [3] 连续 5 轮 写+读 并发 ==");
    begin
      int ok_cnt = 0;
      for (int i = 0; i < 5; i++) begin
        tw='0; tw.write=1; tw.addr=8'h24+i*4; tw.data=32'hA000_0000+i;
        tr='0; tr.write=0; tr.addr=8'h10; tr.data='0;
        fork
          begin host.execute(tw, rw); end
          begin host.execute(tr, rr); end
        join
        if (rw.resp===RESP_OKAY && rr.resp===RESP_OKAY && rr.data===32'h1111_1111)
          ok_cnt++;
      end
      checks++;
      if (ok_cnt == 5) $display("  ✓ 5 轮并发全部正确");
      else begin $display("  ✗ 仅 %0d/5 轮正确", ok_cnt); errors++; end
    end

    $display("------------------------------------------------------");
    $display(" legal result: %0d checks, %0d errors", checks, errors);
    if (errors==0) begin $display("PASS: dual bridge legal concurrency"); $finish; end
    else $fatal(1);
  end
endmodule
