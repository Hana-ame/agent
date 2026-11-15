// ============================================================================
// tb_handshake_protocol_sv.sv：AXI 握手时序规范验证
// ============================================================================
// 本文件记录并验证一条关键规则（本项目多处踩坑后的结论）：
//
//   AXI 握手 = VALID && READY **同拍为高**。
//   响应通道（B/R）上，主站（测试）必须在检测到 VALID 后
//   继续保持 READY，直到从站状态机真正采样到握手并退出。
//
// 反例（错误写法，会导致桥卡死）：
//     if (axi.BVALID) begin b=1; axi.BREADY = 1'b0; end   // ❌ 同拍撤销
//   桥从 S_WR_ACC 进入 S_B_RESP 需要一拍，期间 BVALID 尚未拉高；
//   若测试看到 BVALID 就立即撤 READY，桥在下一沿采样不到握手，
//   会永远卡在 S_B_RESP（state=6）。
//
// 正确写法：保持 READY 直到从站状态机返回 IDLE。
//
// 背景：轻量 UVM 环境的 drive_one 曾犯此错，导致挂死（见 379fec7）。
//       本测试把正确时序固化为可回归的用例。
// ============================================================================
`timescale 1ns/1ps
module tb_handshake_protocol_sv;
  localparam int AW=8,DW=32;
  logic ACLK=0, ARESETn=0;
  always #5 ACLK=~ACLK;
  axi_lite_if #(AW,DW) axi(ACLK);
  apb_if #(AW,DW) apb(ACLK);
  axi_apb_bridge_sv #(AW,DW) bridge(.ACLK(ACLK),.ARESETn(ARESETn),.axi(axi),.apb(apb));
  apb_slave_regs_sv #(AW,DW,16,1) slave(.PCLK(ACLK),.PRESETn(ARESETn),.apb(apb));

  initial begin
    $display("=== AXI 握手时序规范验证 ===");
    ARESETn=0;
    axi.AWVALID=0; axi.WVALID=0; axi.ARVALID=0; axi.BREADY=0; axi.RREADY=0;
    repeat(4) @(posedge ACLK); ARESETn=1; @(posedge ACLK);

    @(posedge ACLK); #1;
    axi.AWVALID=1; axi.AWADDR=8'h10; axi.WVALID=1; axi.WDATA=32'hA5A5;
    axi.BREADY=1;
    begin
      bit aw=0,w=0;
      while (!(aw && w)) begin
        @(posedge ACLK); #1;
        if (!aw && axi.AWREADY) begin aw=1; axi.AWVALID=0; end
        if (!w && axi.WREADY) begin w=1; axi.WVALID=0; end
      end
    end
    // 等桥进入 S_B_RESP 并完成握手（BREADY 全程保持高）
    while (bridge.state != 0) @(posedge ACLK);
    #1;
    axi.BREADY=0;
    $display("  桥已回 IDLE");
    @(posedge ACLK);
    if (bridge.state == 0)
      $display("PASS: AXI handshake protocol (BREADY held until slave exits)");
    else begin
      $display("FAIL: state=%0d", bridge.state);
      $fatal(1);
    end
    $finish;
  end
endmodule
