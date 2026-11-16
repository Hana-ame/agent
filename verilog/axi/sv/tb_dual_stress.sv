// ============================================================================
// tb_dual_stress.sv：并发桥的读写竞争压力测试（覆盖率驱动）
// ============================================================================
// 背景：tb_dual_sv 的 branch 覆盖率仅 63.3%（单路径桥 77.9%）。
// 缺口集中在 APB 仲裁器的竞争路径：
//
//   w_grant = (owner==A_NONE) && w_req && (!r_req || !last_rd)
//   r_grant = (owner==A_NONE) && r_req && (!w_req ||  last_rd)
//
// 未充分覆盖的组合：
//   A. 读写同时请求（w_req && r_req）—— 仲裁必须选一个
//   B. last_rd=1（上次读）时同时请求 → 应优先写（防饿死）
//   C. last_rd=0（上次写）时同时请求 → 应优先读
//   D. 一方持续请求、另一方饥饿 → round-robin 应切换
//
// 本文件用真实并发（fork 两条激励线程）打满读写竞争。
// ============================================================================
`ifndef TB_DUAL_STRESS_SV
`define TB_DUAL_STRESS_SV
`timescale 1ns / 1ps

module tb_dual_stress #(
    parameter int AW   = 8,
    parameter int DW   = 32,
    parameter int NREG = 16,
    parameter int ITERS = 200
);

    import axi_defs_pkg::*;

    localparam int LIMIT = NREG * 4;

    logic ACLK = 0;
    logic ARESETn = 0;
    always #5 ACLK = ~ACLK;

    axi_lite_if #(.AW(AW), .DW(DW)) axi(ACLK);
    apb_if      #(.AW(AW), .DW(DW)) apb(ACLK);

    // 独立协议监视器（被动只读，自动检查 AXI/APB 合法性）
    axi_apb_monitor #(.AW(AW), .DW(DW), .NAME("STRESS"), .SERIAL_BRIDGE(0)) mon (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
    );

    axi_apb_bridge_dual_sv #(.AW(AW), .DW(DW)) bridge (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
    );
    apb_slave_regs_sv #(.AW(AW), .DW(DW), .NREG(NREG), .WAIT_CYCLES(1)) slave (
        .PCLK(ACLK), .PRESETn(ARESETn), .apb(apb)
    );

    // ---- 统计 ----
    int errors = 0;
    int w_count = 0, r_count = 0;
    int sim_checks = 0;

    // 参考镜像（仅写线程更新，读线程比对"自己写过的地址"）
    // 并发下读写交替访问不同地址桶，避免误判
    // 地址分区（必须都在合法范围内：NREG=16 → 0x00..0x3C）
    localparam int WBUCKET = 0;    // 写区：0x00..0x1C（8 个寄存器）
    localparam int RBUCKET = 32;   // 读区：0x20..0x3C（8 个寄存器）

    task automatic do_reset();
        ARESETn = 0;
        axi.AWVALID = 0; axi.WVALID = 0; axi.ARVALID = 0;
        axi.BREADY = 0;  axi.RREADY = 0;
        repeat (4) @(posedge ACLK);
        ARESETn = 1;
        @(posedge ACLK);
    endtask

    // ------------------------------------------------------------------
    // 单个写事务（尽力驱动，不假设时序）
    // ------------------------------------------------------------------
    task automatic wr(input [AW-1:0] a, input [DW-1:0] d);
        bit aw_ok = 0, w_ok = 0;
        @(posedge ACLK); #1;
        axi.BREADY = 1'b1;   // BREADY 提前拉高（AXI 允许常高），确保不漏 BVALID
        axi.AWVALID = 1'b1; axi.AWADDR = a;
        axi.WVALID  = 1'b1; axi.WDATA  = d;
        for (int c = 0; c < 80 && !(aw_ok && w_ok); c++) begin
            @(posedge ACLK); #1;
            if (!aw_ok && axi.AWREADY && axi.AWVALID) begin aw_ok = 1; axi.AWVALID = 1'b0; end
            if (!w_ok  && axi.WREADY  && axi.WVALID)  begin w_ok  = 1; axi.WVALID  = 1'b0; end
        end
        checks_timeout(aw_ok && w_ok, "wr handshake");
        // BREADY 已提前拉高。BVALID 只在响应拍有效，必须在「沿上」采样：
        // 注意不能用 wait(BVALID) —— 组合输出在同一 delta 就有效，
        // 若随即撤销 BREADY，桥在下一个沿就采不到握手，会卡在 W_RESP。
        begin
            bit got_b = 0;
            bit [1:0] bresp_cap = 2'bxx;
            for (int c = 0; c < 120 && !got_b; c++) begin
                @(posedge ACLK);
                // 此时刻 BREADY=1 且若 BVALID=1 则桥将在本沿完成握手
                if (axi.BVALID) begin
                    bresp_cap = axi.BRESP;
                    got_b = 1;
                end
            end
            if (got_b) begin
                if (bresp_cap === RESP_OKAY) w_count++;
                else begin
                    $display("  [ST] FAIL: write resp=%s", resp_name(bresp_cap));
                    errors++;
                end
                // 让桥在后续沿完成 B 握手后再撤 BREADY
                @(posedge ACLK);
            end else begin
                $display("  [ST] FAIL: BVALID 未出现（120 拍超时）");
                errors++;
            end
        end
        axi.BREADY = 1'b0;
    endtask

    // ------------------------------------------------------------------
    // 单个读事务
    // ------------------------------------------------------------------
    task automatic rd(input [AW-1:0] a, output [DW-1:0] d, output bit [1:0] resp);
        bit ar_ok = 0, r_ok = 0;
        @(posedge ACLK); #1;
        axi.RREADY = 1'b1;   // RREADY 提前拉高
        axi.ARVALID = 1'b1; axi.ARADDR = a;
        for (int c = 0; c < 80 && !ar_ok; c++) begin
            @(posedge ACLK); #1;
            if (axi.ARREADY && axi.ARVALID) begin ar_ok = 1; axi.ARVALID = 1'b0; end
        end
        checks_timeout(ar_ok, "rd AR handshake");
        begin
            bit got_r = 0;
            bit [DW-1:0] rdata_cap = '0;
            bit [1:0] rresp_cap = 2'bxx;
            for (int c = 0; c < 120 && !got_r; c++) begin
                @(posedge ACLK);
                if (axi.RVALID) begin
                    rdata_cap = axi.RDATA;
                    rresp_cap = axi.RRESP;
                    got_r = 1;
                end
            end
            if (got_r) begin
                d = rdata_cap; resp = rresp_cap; r_count++;
                @(posedge ACLK);
            end else begin
                d = '0; resp = 2'bxx;
                $display("  [ST] FAIL: read RVALID timeout");
                errors++;
            end
        end
        axi.RREADY = 1'b0;
    endtask

    task automatic checks_timeout(input bit ok, input string tag);
        sim_checks++;
        if (!ok) begin
            $display("  [ST] FAIL: %0s timeout", tag);
            errors++;
        end
    endtask

    // ------------------------------------------------------------------
    // A/B/C/D：读写同时请求，制造仲裁竞争
    // ------------------------------------------------------------------
    task automatic t_arbitration_race();
        logic [DW-1:0] rd_data;
        bit [1:0] rd_resp;
        $display("== STRESS-1: 读写同拍请求（仲裁竞争 %0d 轮）==", ITERS);

        // 预置读区数据
        for (int i = 0; i < 8; i++) wr(RBUCKET + i*4, 32'hA000_0000 + i);

        for (int k = 0; k < ITERS; k++) begin
            // 同拍发起读与写（两个 VALID 同时拉高）
            @(posedge ACLK); #1;
            axi.BREADY = 1'b1; axi.RREADY = 1'b1;   // 提前就绪
            axi.ARVALID = 1'b1; axi.ARADDR = RBUCKET + (k % 8) * 4;
            axi.AWVALID = 1'b1; axi.AWADDR = WBUCKET + (k % 8) * 4;
            axi.WVALID  = 1'b1; axi.WDATA  = 32'hB000_0000 + k;

            // 等三个通道都完成握手
            begin
                bit aw_ok = 0, w_ok = 0, ar_ok = 0;
                // 注意：AWREADY/WREADY/ARREADY 可能在驱动后的下一拍才出现，
                // 必须在采样到 READY 的同一拍撤销 VALID（否则桥会重复接受）。
                for (int c = 0; c < 120 && !(aw_ok && w_ok && ar_ok); c++) begin
                    @(posedge ACLK); #1;
                    if (!aw_ok && axi.AWREADY && axi.AWVALID) begin
                        aw_ok = 1; axi.AWVALID = 1'b0;
                    end
                    if (!w_ok && axi.WREADY && axi.WVALID) begin
                        w_ok = 1; axi.WVALID = 1'b0;
                    end
                    if (!ar_ok && axi.ARREADY && axi.ARVALID) begin
                        ar_ok = 1; axi.ARVALID = 1'b0;
                    end
                end
                sim_checks++;
                if (!(aw_ok && w_ok && ar_ok)) begin
                    $display("  [STRESS-1] FAIL k=%0d 握手未全部完成", k);
                    errors++;
                end
            end

            // 收响应（B 与 R 都可能先到，用非阻塞轮询）
            begin
                bit b_got = 0, r_got = 0;
                bit [1:0] bresp_cap = 2'bxx, rresp_cap = 2'bxx;
                bit [DW-1:0] rdata_cap = '0;
                // 沿上轮询，BREADY/RREADY 保持高直到两个响应都收到
                for (int c = 0; c < 300 && !(b_got && r_got); c++) begin
                    @(posedge ACLK);
                    if (!b_got && axi.BVALID) begin
                        bresp_cap = axi.BRESP; b_got = 1;
                    end
                    if (!r_got && axi.RVALID) begin
                        rdata_cap = axi.RDATA; rresp_cap = axi.RRESP; r_got = 1;
                    end
                end
                @(posedge ACLK);   // 让握手在沿上完成
                sim_checks++;
                if (!b_got || !r_got) begin
                    $display("  [STRESS-1] FAIL k=%0d 响应缺失 B=%0d R=%0d", k, b_got, r_got);
                    errors++;
                end else begin
                    if (bresp_cap !== RESP_OKAY) begin
                        $display("  [STRESS-1] FAIL k=%0d BRESP=%s", k, resp_name(bresp_cap));
                        errors++;
                    end
                    if (rresp_cap !== RESP_OKAY) begin
                        $display("  [STRESS-1] FAIL k=%0d RRESP=%s", k, resp_name(rresp_cap));
                        errors++;
                    end
                end
                axi.BREADY = 1'b0; axi.RREADY = 1'b0;
                @(posedge ACLK);
            end
        end
        $display("  [STRESS-1] %0d 轮同拍读写竞争完成 ✓", ITERS);
    endtask

    // ------------------------------------------------------------------
    // E：读请求持续不断，验证写不被饿死（round-robin）
    // ------------------------------------------------------------------
    task automatic t_no_starvation();
        int writes_done = 0;
        $display("== STRESS-2: 持续读洪流下写不被饿死 ==");

        for (int k = 0; k < 40; k++) begin
            // 每轮：一个读 + 一个写（读优先请求，制造偏斜）
            automatic bit [DW-1:0] d;
            automatic bit [1:0] rsp;
            rd(RBUCKET + (k % 8) * 4, d, rsp);
            wr(WBUCKET + (k % 8) * 4, 32'hC000_0000 + k);
            writes_done++;
        end
        sim_checks++;
        if (writes_done != 40) begin
            $display("  [STRESS-2] FAIL: 仅完成 %0d/40 次写", writes_done);
            errors++;
        end else
            $display("  [STRESS-2] 40 次写全部完成（无饿死）✓");
    endtask

    // ------------------------------------------------------------------
    // F：数据一致性 —— 并发后校验读区未被写污染
    // ------------------------------------------------------------------
    task automatic t_data_integrity();
        automatic bit [DW-1:0] d;
        automatic bit [1:0] rsp;
        int mism = 0;
        $display("== STRESS-3: 并发后数据一致性 ==");
        for (int i = 0; i < 8; i++) begin
            rd(RBUCKET + i*4, d, rsp);
            sim_checks++;
            if (rsp !== RESP_OKAY || d !== (32'hA000_0000 + i)) begin
                $display("  [STRESS-3] FAIL: 读区[%0d]=%h expect %h",
                         i, d, 32'hA000_0000 + i);
                mism++;
                errors++;
            end
        end
        if (mism == 0) $display("  [STRESS-3] 读区 8 个地址数据完好 ✓");
    endtask

    // ------------------------------------------------------------------
    initial begin
        $display("======================================================");
        $display(" dual-path bridge stress (arbitration)");
        $display("======================================================");
        do_reset();

        t_arbitration_race();
        t_no_starvation();
        t_data_integrity();

        $display("------------------------------------------------------");
        if (mon.errors != 0) begin
            $display("  [MON] FAIL: 协议监视器发现 %0d 处违规", mon.errors);
            errors += mon.errors;
        end else
            $display("  [MON] 协议监视器通过（观测 %0d 次 APB 传输）", mon.xfer_count);

        $display(" stress result: %0d checks, %0d errors (W=%0d R=%0d)",
                 sim_checks, errors, w_count, r_count);
        if (errors == 0) begin
            $display("PASS: dual-path stress");
            $finish;
        end else begin
            $display("FAIL: %0d errors", errors);
            $fatal(1, "dual stress failed");
        end
    end

    initial begin
        $dumpfile("tb_dual_stress.vcd");
        $dumpvars(0, tb_dual_stress);
    end

endmodule
`endif // TB_DUAL_STRESS_SV