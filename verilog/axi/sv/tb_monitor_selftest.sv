// 监视器自测：故意制造 APB 违规，确认监视器能抓住
`timescale 1ns/1ps
module tb_mon_selftest;
  localparam int AW=8, DW=32;
  logic ACLK=0, ARESETn=0;
  always #5 ACLK=~ACLK;

  axi_lite_if #(.AW(AW),.DW(DW)) axi(ACLK);
  apb_if      #(.AW(AW),.DW(DW)) apb(ACLK);

  // 被监视的假 APB 从机（由测试直接驱动接口，制造违规）
  axi_apb_monitor #(.AW(AW),.DW(DW), .NAME("SELFTEST"), .VERBOSE(1)) mon(
    .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb));

  int detected = 0;

  initial begin
    // AXI 侧保持空闲（不测 AXI 规则，本测试只验 APB）
    axi.AWVALID=0; axi.WVALID=0; axi.ARVALID=0; axi.BREADY=0; axi.RREADY=0;
    axi.AWADDR=0; axi.WDATA=0; axi.ARADDR=0;
    apb.PSEL=0; apb.PENABLE=0; apb.PWRITE=0; apb.PADDR=0; apb.PWDATA=0;
    apb.PRDATA=0; apb.PREADY=0; apb.PSLVERR=0;

    repeat(3) @(posedge ACLK); ARESETn=1; @(posedge ACLK);
    // 关闭 SVA 断言（本测试只验证监视器；SVA 已被证明独立有效）
    $assertoff(0, tb_mon_selftest);

    // ---- 违规 1：PENABLE=1 但 PSEL=0（APB R1）----
    $display("=== 注入违规 1：PENABLE=1 且 PSEL=0 ===");
    @(posedge ACLK); #1;
    apb.PSEL=0; apb.PENABLE=1;
    @(posedge ACLK); #1;
    apb.PENABLE=0;
    @(posedge ACLK);
    if (mon.errors > detected) begin
      $display("  ✓ 违规 1 被监视器捕获"); detected = mon.errors;
    end else
      $display("  ✗ 违规 1 未被捕获（监视器失效）");

    // ---- 违规 2：SETUP 阶段 PREADY=1（APB R3）----
    $display("=== 注入违规 2：SETUP 阶段 PREADY=1 ===");
    @(posedge ACLK); #1;
    apb.PSEL=1; apb.PENABLE=0; apb.PREADY=1;
    @(posedge ACLK); #1;
    apb.PSEL=0; apb.PREADY=0;
    @(posedge ACLK);
    if (mon.errors > detected) begin
      $display("  ✓ 违规 2 被监视器捕获"); detected = mon.errors;
    end else
      $display("  ✗ 违规 2 未被捕获（监视器失效）");

    // ---- 合法传输：不应报错 ----
    $display("=== 合法 APB 传输（应无新违规）===");
    @(posedge ACLK); #1;
    apb.PSEL=1; apb.PENABLE=0; apb.PWRITE=1; apb.PADDR=8'h00; apb.PWDATA=32'hABCD;
    @(posedge ACLK); #1;
    apb.PENABLE=1;
    @(posedge ACLK); #1;
    apb.PREADY=1;
    @(posedge ACLK); #1;
    apb.PSEL=0; apb.PENABLE=0; apb.PREADY=0;
    @(posedge ACLK);
    if (mon.errors == detected) begin
      $display("  ✓ 合法传输未误报"); 
    end else begin
      $display("  ✗ 合法传输被误报（%0d 新违规）", mon.errors - detected);
      detected = mon.errors;
    end

    $display("------------------------------------------------------");
    $display(" monitor selftest: 捕获 %0d 处违规, 观测 %0d 次传输", mon.errors, mon.xfer_count);
    if (mon.errors >= 2) begin
      $display("PASS: monitor detects violations");
      $finish;
    end else begin
      $display("FAIL: monitor 未能捕获注入的违规");
      $fatal(1);
    end
  end
endmodule
