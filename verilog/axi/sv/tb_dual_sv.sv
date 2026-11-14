// ============================================================================
// tb_dual_sv：验证并发桥（axi_apb_bridge_dual_sv）的读写并行
// ============================================================================
// 验证点：
//   P-06  AW + AR 同拍请求 → 两个 READY 同拍为高（并发接收）
//   D-01  读响应未取走时，写可并行完成（B/R 通道独立）
//   D-02  写响应未取走时，读可并行完成
//   D-04  并发场景数据一致性（读写不串数据）
//
// 握手风格（重要）：
//   所有 VALID/READY 用「posedge 后 #1 采样 + blocking 赋值撤销」，
//   保证握手完成的下一拍 VALID 已为 0 —— 符合 AXI 语义，
//   避免 ``<=`` NBA 延迟一拍造成 SVA 误报。
// ============================================================================
`ifndef TB_DUAL_SV_SV
`define TB_DUAL_SV_SV
`timescale 1ns / 1ps

module tb_dual_sv #(
    parameter int AW = 8,
    parameter int DW = 32,
    parameter int NREG = 16
);

    import axi_defs_pkg::*;

    localparam int LIMIT = NREG * 4;

    logic ACLK = 0;
    logic ARESETn = 0;
    always #5 ACLK = ~ACLK;

    axi_lite_if #(.AW(AW), .DW(DW)) axi(ACLK);
    apb_if      #(.AW(AW), .DW(DW)) apb(ACLK);

    axi_apb_bridge_dual_sv #(.AW(AW), .DW(DW)) bridge (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
    );
    apb_slave_regs_sv #(.AW(AW), .DW(DW), .NREG(NREG), .WAIT_CYCLES(2)) slave (
        .PCLK(ACLK), .PRESETn(ARESETn), .apb(apb)
    );

    int errors = 0;
    int checks = 0;

    task automatic do_reset();
        ARESETn = 0;
        axi.AWVALID = 0; axi.WVALID = 0; axi.ARVALID = 0;
        axi.BREADY = 0;  axi.RREADY = 0;
        repeat (4) @(posedge ACLK);
        ARESETn = 1;
        @(posedge ACLK);
    endtask

    // ------------------------------------------------------------------
    // 基本写事务（AW+W 同拍发，握手后立即撤销 VALID）
    // ------------------------------------------------------------------
    task automatic wr(input [AW-1:0] a, input [DW-1:0] d);
        bit aw_ok = 0, w_ok = 0;
        @(posedge ACLK); #1;
        // 在沿后 #1 驱动，桥在下一个 posedge 采样
        axi.AWVALID = 1'b1; axi.AWADDR = a;
        axi.WVALID  = 1'b1; axi.WDATA  = d;
        // 每拍采样 READY；一旦某通道 READY 为高，本拍末撤销其 VALID
        while (!(aw_ok && w_ok)) begin
            @(posedge ACLK); #1;
            // 注意：这是"沿后"采样，对应的是本拍握手结果
            if (!aw_ok && axi.AWREADY) begin
                aw_ok = 1; axi.AWVALID = 1'b0;
            end
            if (!w_ok && axi.WREADY) begin
                w_ok = 1; axi.WVALID = 1'b0;
            end
            // 若桥已进入 APB 阶段（READY 拉低）但尚未撤销 VALID，主动撤销避免 SVA
            if (!aw_ok && !axi.AWVALID) aw_ok = 1;
            if (!w_ok && !axi.WVALID)  w_ok  = 1;
        end
        axi.BREADY = 1'b1;
        do @(posedge ACLK); while (!axi.BVALID);
        #1; axi.BREADY = 1'b0;
        @(posedge ACLK);
    endtask

    // ------------------------------------------------------------------
    // 基本读事务
    // ------------------------------------------------------------------
    task automatic rd(input [AW-1:0] a, output [DW-1:0] d, output bit [1:0] resp);
        @(posedge ACLK); #1;
        axi.ARVALID = 1'b1; axi.ARADDR = a;
        while (1) begin
            @(posedge ACLK); #1;
            if (axi.ARREADY) begin axi.ARVALID = 1'b0; break; end
        end
        axi.RREADY = 1'b1;
        do @(posedge ACLK); while (!axi.RVALID);
        d = axi.RDATA; resp = axi.RRESP;
        #1; axi.RREADY = 1'b0;
        @(posedge ACLK);
    endtask

    // ------------------------------------------------------------------
    // P-06：AW 与 AR 同拍请求，两个 READY 应同拍为高
    // ------------------------------------------------------------------
    task automatic t_p06_aw_ar_same_cycle();
        bit aw_ok = 0, ar_ok = 0;
        bit aw_ready_seen = 0, ar_ready_seen = 0;
        bit same_cycle = 0;
        $display("== P-06 AW+AR 同拍请求（并发接收）==");
        do_reset();

        // 写数据先就位（W 与 AW 同拍）
        @(posedge ACLK); #1;
        axi.AWVALID = 1'b1; axi.AWADDR = 8'h00;
        axi.WVALID  = 1'b1; axi.WDATA  = 32'hAAAA_0000;
        axi.ARVALID = 1'b1; axi.ARADDR = 8'h04;

        // 逐拍采样，记录 AWREADY/ARREADY 是否在同一拍同时为高
        while (!(aw_ok && ar_ok)) begin
            @(posedge ACLK); #1;
            if (axi.AWREADY && axi.ARREADY && !aw_ok && !ar_ok)
                same_cycle = 1;          // 同一拍两个 READY 都为高
            if (!aw_ok && axi.AWREADY) begin
                aw_ok = 1; aw_ready_seen = 1; axi.AWVALID = 1'b0;
            end
            if (!ar_ok && axi.ARREADY) begin
                ar_ok = 1; ar_ready_seen = 1; axi.ARVALID = 1'b0;
            end
            if (!aw_ok && axi.WREADY) begin
                axi.WVALID = 1'b0;       // W 也接受
            end
        end
        checks++;
        if (!aw_ready_seen) begin
            $display("  [P-06] FAIL: AWREADY 未出现"); errors++;
        end
        if (!ar_ready_seen) begin
            $display("  [P-06] FAIL: ARREADY 未出现"); errors++;
        end
        $display("  [P-06] AW/AR 同拍 ready = %0d（1=并发接受）", same_cycle);

        // 收响应
        axi.BREADY = 1'b1; axi.RREADY = 1'b1;
        // 分别取走 B 与 R
        while (!axi.BVALID) @(posedge ACLK);
        while (!axi.RVALID) @(posedge ACLK);
        do @(posedge ACLK); while (axi.BVALID || axi.RVALID);
        #1; axi.BREADY = 1'b0; axi.RREADY = 1'b0;
        @(posedge ACLK);
    endtask

    // ------------------------------------------------------------------
    // D-01：读响应未取走时写可完成
    // ------------------------------------------------------------------
    task automatic t_d01_b_during_r();
        bit aw_ok = 0, w_ok = 0;
        $display("== D-01 R 未取走时 B 可完成 ==");
        do_reset();
        // 预置数据
        wr(8'h08, 32'h1234_5678);

        // 发起读，等 RVALID 但保持 RREADY=0
        @(posedge ACLK); #1;
        axi.ARVALID = 1'b1; axi.ARADDR = 8'h08;
        while (1) begin
            @(posedge ACLK); #1;
            if (axi.ARREADY) begin axi.ARVALID = 1'b0; break; end
        end
        do @(posedge ACLK); while (!axi.RVALID);
        checks++;
        if (axi.RDATA !== 32'h1234_5678) begin
            $display("  [D-01] FAIL: RDATA=%h expect 12345678", axi.RDATA); errors++;
        end

        // R 未取走，发写，应能完成
        @(posedge ACLK); #1;
        axi.AWVALID = 1'b1; axi.AWADDR = 8'h0C;
        axi.WVALID  = 1'b1; axi.WDATA  = 32'hCAFE_9999;
        while (!(aw_ok && w_ok)) begin
            @(posedge ACLK); #1;
            if (!aw_ok && axi.AWREADY) begin aw_ok = 1; axi.AWVALID = 1'b0; end
            if (!w_ok  && axi.WREADY)  begin w_ok  = 1; axi.WVALID  = 1'b0; end
            if (!aw_ok && !axi.AWVALID) aw_ok = 1;
            if (!w_ok  && !axi.WVALID)  w_ok  = 1;
        end
        axi.BREADY = 1'b1;
        do @(posedge ACLK); while (!axi.BVALID);
        checks++;
        $display("  [D-01] B 在 R 未取走时完成 ✓");
        #1; axi.BREADY = 1'b0;
        // 现在取走 R（RREADY 保持到握手完成）
        axi.RREADY = 1'b1;
        do @(posedge ACLK); while (axi.RVALID);
        #1; axi.RREADY = 1'b0;
        @(posedge ACLK);
    endtask

    // ------------------------------------------------------------------
    // D-02：写响应未取走时读可完成
    // ------------------------------------------------------------------
    task automatic t_d02_r_during_b();
        bit aw_ok = 0, w_ok = 0;
        $display("== D-02 B 未取走时 R 可完成 ==");
        do_reset();
        wr(8'h10, 32'h0BAD_F00D);

        // 发起写，等 BVALID 但保持 BREADY=0
        @(posedge ACLK); #1;
        axi.AWVALID = 1'b1; axi.AWADDR = 8'h14;
        axi.WVALID  = 1'b1; axi.WDATA  = 32'h7777_7777;
        while (!(aw_ok && w_ok)) begin
            @(posedge ACLK); #1;
            if (!aw_ok && axi.AWREADY) begin aw_ok = 1; axi.AWVALID = 1'b0; end
            if (!w_ok  && axi.WREADY)  begin w_ok  = 1; axi.WVALID  = 1'b0; end
            if (!aw_ok && !axi.AWVALID) aw_ok = 1;
            if (!w_ok  && !axi.WVALID)  w_ok  = 1;
        end
        while (!axi.BVALID) begin @(posedge ACLK); #1; end

        // B 未取走，发读，应能完成
        @(posedge ACLK); #1;
        axi.ARVALID = 1'b1; axi.ARADDR = 8'h10;
        while (1) begin
            @(posedge ACLK); #1;
            if (axi.ARREADY) begin axi.ARVALID = 1'b0; break; end
        end
        axi.RREADY = 1'b1;
        do @(posedge ACLK); while (!axi.RVALID);
        checks++;
        if (axi.RDATA !== 32'h0BAD_F00D) begin
            $display("  [D-02] FAIL: RDATA=%h expect 0BADF00D", axi.RDATA); errors++;
        end else
            $display("  [D-02] R 在 B 未取走时完成 ✓");
        #1; axi.RREADY = 1'b0;
        axi.BREADY = 1'b1;
        do @(posedge ACLK); while (axi.BVALID);
        #1; axi.BREADY = 1'b0;
        @(posedge ACLK);
    endtask

    // ------------------------------------------------------------------
    // D-04：并发场景数据一致性
    // ------------------------------------------------------------------
    task automatic t_d04_consistency();
        bit [DW-1:0] mirror [0:NREG-1];
        $display("== D-04 并发数据一致性 ==");
        do_reset();
        for (int i = 0; i < NREG; i++) mirror[i] = '0;

        for (int k = 0; k < 30; k++) begin
            automatic int idx = $urandom % NREG;
            automatic bit [DW-1:0] d = $urandom;
            automatic int ridx = $urandom % NREG;
            automatic bit [DW-1:0] got;
            automatic bit [1:0] resp;

            wr(idx*4, d);
            mirror[idx] = d;

            rd(ridx*4, got, resp);
            checks++;
            if (resp !== RESP_OKAY || got !== mirror[ridx]) begin
                $display("  [D-04] FAIL k=%0d rd idx=%0d exp=%h got=%h resp=%s",
                         k, ridx, mirror[ridx], got, resp_name(resp));
                errors++;
            end
        end
    endtask

    // ------------------------------------------------------------------
    initial begin
        $display("======================================================");
        $display(" dual-path bridge concurrency test");
        $display("======================================================");
        do_reset();
        t_p06_aw_ar_same_cycle();
        t_d01_b_during_r();
        t_d02_r_during_b();
        t_d04_consistency();

        $display("------------------------------------------------------");
        $display(" dual result: %0d checks, %0d errors", checks, errors);
        if (errors == 0) begin
            $display("PASS: dual-path bridge concurrency");
            $finish;
        end else begin
            $display("FAIL: %0d errors", errors);
            $fatal(1, "dual bridge failed");
        end
    end

    initial begin
        $dumpfile("tb_dual_sv.vcd");
        $dumpvars(0, tb_dual_sv);
    end

endmodule
`endif // TB_DUAL_SV_SV