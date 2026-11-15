// 绕过 txn_t，直接用 AXI 接口验证 DW=64 时桥/从机是否正确
// ============================================================================
// tb_dw64_sv.sv：DW=64 数据通路验证（绕过 txn_t 的 32 位限制）
// ============================================================================
// 背景：参数矩阵测试发现 DW=64 时数据被截断到 32 位，根因是
// txn_t.data / rsp_t.data 硬编码 32 位（验证基础设施限制，非 RTL 缺陷）。
//
// 该限制已修复：bus_txn_pkg.sv 的数据位宽改由 TXN_DW 宏配置
// （默认 32，可 -DTXN_DW=64 切换），params 目标会自动按 DW 传入。
// 因此 tb_param_sv 现在可以在 DW=64 下做完整数据检查。
//
// 本测试保留为「绕过事务抽象层」的独立通路验证：
// 直接驱动 AXI 接口，确保 64 位数据在桥与从机之间无损传输，
// 不依赖 txn_t 的位宽配置。
// ============================================================================
`timescale 1ns/1ps
module tb_dw64_sv;
  localparam int AW=10, DW=64, NREG=16;
  logic ACLK=0, ARESETn=0;
  always #5 ACLK=~ACLK;
  axi_lite_if #(.AW(AW),.DW(DW)) axi(ACLK);
  apb_if      #(.AW(AW),.DW(DW)) apb(ACLK);
  axi_apb_bridge_sv #(.AW(AW),.DW(DW)) bridge(
    .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb));
  apb_slave_regs_sv #(.AW(AW),.DW(DW),.NREG(NREG),.WAIT_CYCLES(2)) slave(
    .PCLK(ACLK), .PRESETn(ARESETn), .apb(apb));

  int errors=0, checks=0;
  logic [DW-1:0] got;
  bit [1:0] resp;

  task automatic wr(input [AW-1:0] a, input [DW-1:0] d);
    @(posedge ACLK);
    axi.AWVALID <= 1; axi.AWADDR <= a;
    axi.WVALID <= 1; axi.WDATA <= d;
    axi.BREADY <= 1;
    begin
      bit aw=0,w=0;
      while (!(aw&&w)) begin
        @(posedge ACLK);
        if (!aw && axi.AWREADY) begin aw=1; axi.AWVALID<=0; end
        if (!w && axi.WREADY) begin w=1; axi.WVALID<=0; end
      end
    end
    do @(posedge ACLK); while (!axi.BVALID);
    resp = axi.BRESP;
    @(posedge ACLK); axi.BREADY<=0;
    @(posedge ACLK);
  endtask

  task automatic rd(input [AW-1:0] a);
    @(posedge ACLK);
    axi.ARVALID <= 1; axi.ARADDR <= a; axi.RREADY <= 1;
    while (1) begin
      @(posedge ACLK);
      if (axi.ARREADY) begin axi.ARVALID<=0; break; end
    end
    do @(posedge ACLK); while (!axi.RVALID);
    got = axi.RDATA; resp = axi.RRESP;
    @(posedge ACLK); axi.RREADY<=0;
    @(posedge ACLK);
  endtask

  initial begin
    $display("=== DW=64 数据通路验证（绕过 txn_t）===");
    ARESETn=0;
    axi.AWVALID<=0; axi.WVALID<=0; axi.ARVALID<=0; axi.BREADY<=0; axi.RREADY<=0;
    repeat(4) @(posedge ACLK); ARESETn=1; @(posedge ACLK);

    wr(10'h000, 64'h1122334455667788);
    rd(10'h000);
    checks++;
    if (got !== 64'h1122334455667788) begin
      $display("  ✗ DW=64 读写数据错误: got=%h", got); errors++;
    end else $display("  ✓ DW=64 读写数据完整（1122334455667788）");

    wr(10'h03C, 64'hAABBCCDDEEFF0011);  // 最后一字
    rd(10'h03C);
    checks++;
    if (got !== 64'hAABBCCDDEEFF0011) begin
      $display("  ✗ DW=64 末地址数据错误: got=%h", got); errors++;
    end else $display("  ✓ DW=64 末地址数据完整");

    wr(10'h040, 64'h1234);  // 越界
    checks++;
    if (resp !== 2'b10) begin
      $display("  ✗ DW=64 越界未返回 SLVERR: resp=%b", resp); errors++;
    end else $display("  ✓ DW=64 越界返回 SLVERR");

    $display("--------------------------------------------------");
    if (errors==0) begin $display("PASS: DW=64 桥与从机功能正确"); $finish; end
    else begin $display("FAIL: %0d errors", errors); $fatal(1); end
  end
endmodule
