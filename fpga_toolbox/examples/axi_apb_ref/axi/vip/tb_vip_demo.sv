// ============================================================================
// tb_vip_demo：证明 VIP 可直接接入现有 DUT（三行接入）
// ============================================================================
// DUT 侧不变：axi_apb_bridge → apb_if → apb_slave_regs
// VIP 侧只做三件事：setup / run_sequence / report
`ifndef TB_VIP_DEMO_SV
`define TB_VIP_DEMO_SV
`timescale 1ns / 1ps

module tb_vip_demo;

    import bus_txn_pkg::*;
    import axi_defs_pkg::*;
    import axi_vip_pkg::*;

    localparam int AW = 8;
    localparam int DW = 32;
    localparam int NREG = 16;

    logic ACLK = 0;
    logic ARESETn = 0;
    always #5 ACLK = ~ACLK;

    axi_lite_if #(.AW(AW), .DW(DW)) axi(ACLK);
    apb_if      #(.AW(AW), .DW(DW)) apb(ACLK);

    // ---- DUT（与之前完全相同的链路，未做任何为 VIP 的改动）----
    axi_apb_bridge_sv #(.AW(AW), .DW(DW)) bridge (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
    );
    apb_slave_regs_sv #(.AW(AW), .DW(DW), .NREG(NREG), .WAIT_CYCLES(2)) slave (
        .PCLK(ACLK), .PRESETn(ARESETn), .apb(apb)
    );

    // ---- VIP：三行接入 ----
    axi_lite_vip #(.AW(AW), .DW(DW)) vip (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi)
    );

    initial begin
        automatic axi_sequence seq = new();
        // 1) 配置
        vip.setup(.limit(NREG*4), .vb(0));

        // 复位
        repeat (4) @(posedge ACLK);
        ARESETn = 1;
        $display("== VIP demo: AXI->APB->AXI (NREG=%0d) ==", NREG);

        // 2) 构造序列并执行
        seq.add_loopback(8'h00, 32'hDEAD_BEEF, 8'd1);   // 写后读回（pause=1）
        seq.add_loopback(8'h04, 32'h1234_5678, 8'd0);   // 背靠背
        seq.add(1'b1, 8'h08, 32'hCAFE_F00D, 8'd3, 1'b1); // 同拍写 + pause=3
        seq.add_random(60, NREG*4);                     // 60 笔随机
        seq.add(1'b1, NREG*4, 32'hBAD_0BAD, 8'd1, 1'b0, 1'b1); // 越界写 → 期望 SLVERR
        seq.add(1'b0, NREG*4, '0, 8'd1, 1'b0, 1'b1);           // 越界读 → 期望 SLVERR

        vip.run_sequence(seq);
        vip.execute_all();

        // 3) 报告
        if (!vip.report()) $fatal(1, "VIP FAILED");
        $finish;
    end

    initial begin
        $dumpfile("tb_vip_demo.vcd");
        $dumpvars(0, tb_vip_demo);
    end

endmodule
`endif // TB_VIP_DEMO_SV