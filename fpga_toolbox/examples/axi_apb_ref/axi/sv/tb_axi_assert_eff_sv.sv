// ============================================================================
// tb_axi_assert_eff_sv.sv：AXI 5 条断言的有效性验证
// ============================================================================
// 上轮发现这 5 条断言因前件从不成立而「真空通过」；
// 本轮进一步验证：注入违规时它们是否真的会失败。
//
// 每条断言单独构建（-DACASE=<1..5>），注入对应违规：
//   1  AWVALID 未握手即撤销  → ap_aw_valid_stable
//   2  WVALID  未握手即撤销  → ap_w_valid_stable
//   3  ARVALID 未握手即撤销  → ap_ar_valid_stable
//   4  BVALID  未握手即撤销  → ap_b_valid_stable
//   5  RVALID  未握手即撤销  → ap_r_valid_stable
//   0  无违规（应无断言失败）
//
// 驱动要点：必须用非阻塞赋值 + 沿后驱动，否则 SVA 采样不到前件
//（首版用阻塞赋值直接驱动，5 条断言全部未捕获，排查后确认是驱动时序问题）。
// ============================================================================
`timescale 1ns/1ps
`ifndef ACASE
  `define ACASE 0
`endif
module tb_axi_assert_eff_sv;
  localparam int AW=8, DW=32;
  logic ACLK=0, ARESETn=0;
  always #5 ACLK=~ACLK;
  axi_lite_if #(.AW(AW),.DW(DW)) axi(ACLK);

  initial begin
    $display("=== AXI 断言有效性 ACASE=%0d ===", `ACASE);
    ARESETn=0;
    axi.AWVALID=0; axi.AWADDR=0; axi.AWREADY=0;
    axi.WVALID=0; axi.WDATA=0; axi.WREADY=0;
    axi.ARVALID=0; axi.ARADDR=0; axi.ARREADY=0;
    axi.BVALID=0; axi.BRESP=0; axi.BREADY=0;
    axi.RVALID=0; axi.RDATA=0; axi.RRESP=0; axi.RREADY=0;
    repeat(3) @(posedge ACLK); ARESETn=1; @(posedge ACLK);

    case (`ACASE)
      1: begin  // 违反 AWVALID 保持
        // 用 NBA 驱动，确保信号在时钟沿稳定被采样
        @(posedge ACLK); #1;
        axi.AWVALID <= 1; axi.AWADDR <= 8'h0;
        @(posedge ACLK); #1;   // 此时 SVA 看到 AWVALID=1 && !AWREADY
        axi.AWVALID <= 0;      // 下一沿撤销 → 违反
        @(posedge ACLK); #1;
        @(posedge ACLK);
      end
      2: begin
        @(posedge ACLK); #1; axi.WVALID <= 1; axi.WDATA <= 32'h1;
        @(posedge ACLK); #1; axi.WVALID <= 0;
        @(posedge ACLK); #1; @(posedge ACLK);
      end
      3: begin
        @(posedge ACLK); #1; axi.ARVALID <= 1; axi.ARADDR <= 8'h0;
        @(posedge ACLK); #1; axi.ARVALID <= 0;
        @(posedge ACLK); #1; @(posedge ACLK);
      end
      4: begin
        @(posedge ACLK); #1; axi.BVALID <= 1; axi.BRESP <= 2'b00;
        @(posedge ACLK); #1; axi.BVALID <= 0;
        @(posedge ACLK); #1; @(posedge ACLK);
      end
      5: begin
        @(posedge ACLK); #1; axi.RVALID <= 1; axi.RDATA <= 32'h1; axi.RRESP <= 2'b00;
        @(posedge ACLK); #1; axi.RVALID <= 0;
        @(posedge ACLK); #1; @(posedge ACLK);
      end
      default: begin
        $display("  (无违规，应无断言失败)");
        @(posedge ACLK); @(posedge ACLK);
        $display("PASS: clean");
        $finish;
      end
    endcase
    $display("FAIL: 断言未捕获 ACASE=%0d 违规", `ACASE);
    $fatal(1);
  end
endmodule
