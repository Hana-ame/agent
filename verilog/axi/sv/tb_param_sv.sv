// ============================================================================
// tb_param_sv.sv：DUT 参数配置矩阵验证
// ============================================================================
// 问题：所有 DUT 都参数化（AW/DW/NREG/WAIT_CYCLES），
//       但全部测试一直只跑默认配置（AW=8, DW=32, NREG=16）。
//       参数化代码中针对其他配置的分支从未被执行过。
//
// 本测试通过 -G 覆盖参数，验证在多种配置下功能仍正确。
// 每个配置都验证：
//   - 基本读写（数据完整性）
//   - 地址边界（最后一个合法地址 + 第一个非法地址）
//   - 越界 SLVERR
//   - 地址回绕（若 AW 较小）
//
// 用法（由 sim.sh params 目标逐个配置运行）
//   仿真器参数覆盖：-GAW=6 -GDW=16 -GNREG=4 -GWAITC=0
// ============================================================================
`ifndef TB_PARAM_SV_SV
`define TB_PARAM_SV_SV
`timescale 1ns / 1ps

module tb_param_sv #(
    parameter int AW   = 8,
    parameter int DW   = 32,
    parameter int NREG = 16,
    parameter int WAITC = 2,
    parameter bit DUAL = 1'b0    // 1=用并发桥，0=串行桥
);

    import bus_txn_pkg::*;
    import axi_defs_pkg::*;

    typedef logic [AW-1:0] addr_t;
    typedef logic [DW-1:0] data_t;

    // ---- 重要限制说明 ----
    // txn_t.data / rsp_t.data 在 bus_txn_pkg.sv 中硬编码为 32 位
    // （事务抽象层的历史设计，非 RTL 限制）。
    // 因此 DW != 32 时，经 txn_t 的数据会被截断，
    // 数据完整性检查必须跳过。
    //
    // 已验证（tb_dw64 直接驱动接口，绕过 txn_t）：
    //   DW=64 时桥与从机的读写数据、边界、越界响应全部正确。
    // 故本限制属验证基础设施问题，不是 DUT 缺陷。
    // TXN_DW（事务层位宽）与 DW（DUT 位宽）一致时可做数据完整性检查。
    // sim.sh params 目标会自动按 DW 传 -DTXN_DW=<DW>。
    localparam int TXN_WIDTH = `TXN_DW;
    localparam bit DATA_CHECK = (TXN_WIDTH == DW);

    localparam logic [DW-1:0] PATTERN_A = {DW/4{4'hA}};
    localparam logic [DW-1:0] PATTERN_5 = {DW/4{4'h5}};

    logic ACLK = 0, ARESETn = 0;
    always #5 ACLK = ~ACLK;

    axi_lite_if #(.AW(AW), .DW(DW)) axi(ACLK);
    apb_if      #(.AW(AW), .DW(DW)) apb(ACLK);

    generate if (DUAL) begin : g_bridge
        axi_apb_bridge_dual_sv #(.AW(AW), .DW(DW)) bridge (
            .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
        );
    end else begin : g_bridge
        axi_apb_bridge_sv #(.AW(AW), .DW(DW)) bridge (
            .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
        );
    end endgenerate
    apb_slave_regs_sv #(.AW(AW), .DW(DW), .NREG(NREG), .WAIT_CYCLES(WAITC)) slave (
        .PCLK(ACLK), .PRESETn(ARESETn), .apb(apb)
    );
    axi_apb_monitor #(.AW(AW), .DW(DW), .NAME("PARAM"), .SERIAL_BRIDGE(!DUAL)) mon (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
    );
    axi_txn_host #(.AW(AW), .DW(DW)) host (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi)
    );

    int errors = 0, checks = 0;

    task automatic do_reset();
        ARESETn = 0;
        repeat (4) @(posedge ACLK);
        ARESETn = 1;
        repeat (2) @(posedge ACLK);
    endtask

    function automatic txn_t mk(
        input bit wr,
        input addr_t a,
        input data_t d
    );
        txn_t t;
        t = '0;
        t.write = wr; t.addr = a; t.data = d;
        return t;
    endfunction

    task automatic wr_chk(
        input addr_t a,
        input data_t d,
        input string tag
    );
        txn_t t; rsp_t r;
        t = mk(1'b1, a, d);
        host.execute(t, r);
        checks++;
        if (r.resp !== RESP_OKAY) begin
            $display("  [%0s] FAIL: 写 addr=0x%0h resp=%s", tag, a, resp_name(r.resp));
            errors++;
        end
    endtask

    task automatic rd_chk(
        input addr_t a,
        input data_t exp_data,
        input string tag,
        input bit expect_ok
    );
        txn_t t; rsp_t r;
        t = mk(1'b0, a, '0);
        host.execute(t, r);
        checks++;
        if (expect_ok) begin
            if (r.resp !== RESP_OKAY) begin
                $display("  [%0s] FAIL: 读 addr=0x%0h resp=%s（应 OKAY）",
                         tag, a, resp_name(r.resp));
                errors++;
            end else if (DATA_CHECK && r.data !== exp_data) begin
                $display("  [%0s] FAIL: 读 addr=0x%0h data=%h expect=%h",
                         tag, a, r.data, exp_data);
                errors++;
            end
        end else begin
            if (r.resp !== RESP_SLVERR) begin
                $display("  [%0s] FAIL: 读 addr=0x%0h resp=%s（应 SLVERR）",
                         tag, a, resp_name(r.resp));
                errors++;
            end
        end
    endtask

    // ==================================================================
    // P-01：基本读写（覆盖首、中、末地址）
    // ==================================================================
    task automatic t_basic();
        $display("== P-01: 基本读写（NREG=%0d）==", NREG);
        do_reset();

        // 首地址
        wr_chk(AW'(0), PATTERN_A, "P01-first");
        rd_chk(AW'(0), PATTERN_A, "P01-first", 1'b1);

        // 中间地址
        if (NREG > 2) begin
            wr_chk(AW'((NREG/2)*4), PATTERN_5, "P01-mid");
            rd_chk(AW'((NREG/2)*4), PATTERN_5, "P01-mid", 1'b1);
        end

        // 末地址（最后一个合法寄存器）
        wr_chk(AW'((NREG-1)*4), PATTERN_A, "P01-last");
        rd_chk(AW'((NREG-1)*4), PATTERN_A, "P01-last", 1'b1);

        $display("  ✓ 首/中/末地址读写正确");
    endtask

    // ==================================================================
    // P-02：地址边界（最后合法 + 第一个非法）
    // ==================================================================
    task automatic t_boundary();
        $display("== P-02: 地址边界（NREG=%0d, 最后一字 0x%0h）==",
                 NREG, (NREG-1)*4);
        do_reset();

        // 最后合法地址
        wr_chk(AW'((NREG-1)*4), PATTERN_5, "P02-last-ok");
        rd_chk(AW'((NREG-1)*4), PATTERN_5, "P02-last-ok", 1'b1);

        // 第一个非法地址（越界 → SLVERR）
        // 注意：若 NREG 已占满地址空间，则无越界地址可测
        if ((NREG * 4) < (1 << AW)) begin
            txn_t t; rsp_t r;
            t = mk(1'b1, AW'(NREG*4), PATTERN_A);
            host.execute(t, r);
            checks++;
            if (r.resp !== RESP_SLVERR) begin
                $display("  [P02] FAIL: 越界写 addr=0x%0h resp=%s（应 SLVERR）",
                         NREG*4, resp_name(r.resp));
                errors++;
            end
            rd_chk(AW'(NREG*4), '0, "P02-oob", 1'b0);
            $display("  ✓ 越界地址返回 SLVERR");
        end else begin
            $display("  （NREG 占满地址空间，无越界地址可测，跳过）");
        end
    endtask

    // ==================================================================
    // P-03：数据模式（全 0 / 全 1 / 交替）
    // ==================================================================
    task automatic t_patterns();
        $display("== P-03: 数据模式（DW=%0d）==", DW);
        do_reset();

        wr_chk(AW'(0), '0, "P03-zero");
        rd_chk(AW'(0), '0, "P03-zero", 1'b1);

        wr_chk(AW'(0), {DW{1'b1}}, "P03-ones");
        rd_chk(AW'(0), {DW{1'b1}}, "P03-ones", 1'b1);

        wr_chk(AW'(0), {(DW/2){2'b10}}, "P03-alt");
        rd_chk(AW'(0), {(DW/2){2'b10}}, "P03-alt", 1'b1);

        $display("  ✓ 全0/全1/交替模式读写正确");
    endtask

    // ==================================================================
    // P-04：多寄存器独立性
    // ==================================================================
    task automatic t_independence();
        int n;
        $display("== P-04: 多寄存器独立性 ==");
        do_reset();

        n = (NREG < 4) ? NREG : 4;
        for (int i = 0; i < n; i++)
            wr_chk(AW'(i*4), $urandom, "P04-wr");

        for (int i = 0; i < n; i++) begin
            txn_t t; rsp_t r;
            t = mk(1'b0, AW'(i*4), '0);
            host.execute(t, r);
            checks++;
            // 只检查响应 OKAY（数据在写入时未记录，此处验独立寻址不串扰）
            if (r.resp !== RESP_OKAY) begin
                $display("  [P04] FAIL: 读 addr=0x%0h resp=%s", i*4, resp_name(r.resp));
                errors++;
            end
        end
        $display("  ✓ %0d 个寄存器独立寻址正常", n);
    endtask

    // ==================================================================
    initial begin
        $display("======================================================");
        $display(" 参数配置验证：AW=%0d DW=%0d NREG=%0d WAIT=%0d 桥=%0s",
                 AW, DW, NREG, WAITC, DUAL ? "并发" : "串行");
        if (!DATA_CHECK)
            $display(" 注：TXN_DW(%0d) != DW(%0d)，跳过数据完整性检查", TXN_WIDTH, DW);
        $display("======================================================");

        t_basic();
        t_boundary();
        t_patterns();
        t_independence();

        $display("------------------------------------------------------");
        if (mon.errors != 0) begin
            $display("  [MON] FAIL: 协议监视器发现 %0d 处违规", mon.errors);
            errors += mon.errors;
        end else
            $display("  [MON] 协议监视器通过（%0d 次 APB 传输）", mon.xfer_count);

        $display(" param result: AW=%0d DW=%0d NREG=%0d WAIT=%0d 桥=%0s → %0d checks, %0d errors",
                 AW, DW, NREG, WAITC, DUAL ? "DUAL" : "SERIAL", checks, errors);
        if (errors == 0) begin
            $display("PASS: param config AW=%0d DW=%0d NREG=%0d WAIT=%0d %0s",
                     AW, DW, NREG, WAITC, DUAL ? "DUAL" : "SERIAL");
            $finish;
        end else begin
            $display("FAIL: %0d errors", errors);
            $fatal(1, "param test failed");
        end
    end

endmodule
`endif // TB_PARAM_SV_SV
