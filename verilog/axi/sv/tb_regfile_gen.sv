// ============================================================================
// tb_regfile_gen：属性验证（由 gen_regfile.py 从 regfile_spec.json 生成）
// ============================================================================
// ⚠️ 自动生成，勿手工编辑。
//
// 覆盖寄存器（12 个）：
    //   0x00  VERSION      RO    reset=0x00010203
    //   0x04  INT_STATUS   W1C   reset=0x000000F3
    //   0x08  RESERVED0    RSVD  reset=0x00000000
    //   0x0C  CTRL         RW    reset=0x00000000
    //   0x10  STATUS       RO    reset=0x00000011
    //   0x14  INT_ENABLE   RW    reset=0x00000000
    //   0x18  DATA_IN      RW    reset=0x00000000
    //   0x1C  DATA_OUT     RW    reset=0x00000000
    //   0x20  SCRATCH      RW    reset=0x00000000
    //   0x24  IRQ_CLEAR    W1C   reset=0x0000000F
    //   0x28  INT_SET      W1S   reset=0x00000000
    //   0x2C  EVENT_CNT    RC    reset=0x000000A5
// ============================================================================
`ifndef TB_REGFILE_GEN_SV
`define TB_REGFILE_GEN_SV
`timescale 1ns / 1ps

module tb_regfile_gen #(
    parameter bit DUAL = 1'b0,   // 1=并发桥
    parameter int AW   = 8,
    parameter int DW   = 32,
    parameter int NREG = 12
);

    import bus_txn_pkg::*;
    import axi_defs_pkg::*;

    logic ACLK = 0, ARESETn = 0;
    always #5 ACLK = ~ACLK;

    axi_lite_if #(.AW(AW), .DW(DW)) axi(ACLK);
    apb_if      #(.AW(AW), .DW(DW)) apb(ACLK);

    generate if (DUAL) begin : g_br
        axi_apb_bridge_dual_sv #(.AW(AW), .DW(DW)) bridge (
            .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb));
    end else begin : g_br
        axi_apb_bridge_sv #(.AW(AW), .DW(DW)) bridge (
            .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb));
    end endgenerate
    apb_slave_regfile_gen #(.AW(AW), .DW(DW), .NREG(NREG), .WAIT_CYCLES(2)) dut (
        .PCLK(ACLK), .PRESETn(ARESETn), .apb(apb)
    );
    axi_txn_host #(.AW(AW), .DW(DW)) host (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi)
    );

    int errors = 0;
    int checks = 0;

    task automatic do_reset();
        ARESETn = 0;
        repeat (4) @(posedge ACLK);
        ARESETn = 1;
        repeat (2) @(posedge ACLK);
    endtask

    function automatic txn_t mk(input bit wr, input [AW-1:0] a, input [DW-1:0] d);
        txn_t t;
        t = '0;
        t.write = wr; t.addr = a; t.data = d;
        return t;
    endfunction

    task automatic write_reg(input [AW-1:0] a, input [DW-1:0] d);
        txn_t t; rsp_t r;
        t = mk(1'b1, a, d);
        host.execute(t, r);
        checks++;
        if (r.resp !== RESP_OKAY) begin
            $display("  [TB] FAIL: 写 addr=%0h resp=%s", a, resp_name(r.resp));
            errors++;
        end
    endtask

    task automatic read_reg(input [AW-1:0] a, output [DW-1:0] d);
        txn_t t; rsp_t r;
        t = mk(1'b0, a, '0);
        host.execute(t, r);
        checks++;
        d = r.data;
        if (r.resp !== RESP_OKAY) begin
            $display("  [TB] FAIL: 读 addr=%0h resp=%s", a, resp_name(r.resp));
            errors++;
        end
    endtask

    logic [DW-1:0] d, wv, base;

    initial begin
        $display("======================================================");
        $display(" generated regfile test (12 registers)");
        $display("======================================================");
        do_reset();

        $display("== RO 寄存器（只读）==");
        // VERSION: 初始 0x00010203
        read_reg(8'h00, d);
        checks++;
        if (d !== 32'h00010203) begin
            $display("  [RO] FAIL: VERSION 初值=%h expect 00010203", d);
            errors++;
        end
        write_reg(8'h00, 32'hFFFF_FFFF);   // 应被忽略
        read_reg(8'h00, d);
        checks++;
        if (d !== 32'h00010203) begin
            $display("  [RO] FAIL: VERSION 被写改动=%h", d);
            errors++;
        end
        // STATUS: 初始 0x00000011
        read_reg(8'h10, d);
        checks++;
        if (d !== 32'h00000011) begin
            $display("  [RO] FAIL: STATUS 初值=%h expect 00000011", d);
            errors++;
        end
        write_reg(8'h10, 32'hFFFF_FFFF);   // 应被忽略
        read_reg(8'h10, d);
        checks++;
        if (d !== 32'h00000011) begin
            $display("  [RO] FAIL: STATUS 被写改动=%h", d);
            errors++;
        end

        $display("== W1C 寄存器（写 1 清零）==");
        // INT_STATUS: 复位值 0x000000F3，写全 1 应清零
        read_reg(8'h04, d);
        checks++;
        if (d !== 32'h000000F3) begin
            $display("  [W1C] FAIL: INT_STATUS 复位值=%h expect 000000F3", d);
            errors++;
        end
        write_reg(8'h04, 32'hFFFF_FFFF);
        read_reg(8'h04, d);
        checks++;
        if (d !== '0) begin
            $display("  [W1C] FAIL: INT_STATUS 写 1 后应 0，实际=%h", d);
            errors++;
        end
        // IRQ_CLEAR: 复位值 0x0000000F，写全 1 应清零
        read_reg(8'h24, d);
        checks++;
        if (d !== 32'h0000000F) begin
            $display("  [W1C] FAIL: IRQ_CLEAR 复位值=%h expect 0000000F", d);
            errors++;
        end
        write_reg(8'h24, 32'hFFFF_FFFF);
        read_reg(8'h24, d);
        checks++;
        if (d !== '0) begin
            $display("  [W1C] FAIL: IRQ_CLEAR 写 1 后应 0，实际=%h", d);
            errors++;
        end

        $display("== RSVD 寄存器（保留）==");
        // RESERVED0: 读恒 0
        write_reg(8'h08, 32'hDEAD_BEEF);
        read_reg(8'h08, d);
        checks++;
        if (d !== '0) begin
            $display("  [RSVD] FAIL: RESERVED0 读应 0，实际=%h", d);
            errors++;
        end

        $display("== RW 寄存器（读写）==");
        // INT_ENABLE: 写随机值读回一致
        wv = $urandom;
        write_reg(8'h14, wv);
        read_reg(8'h14, d);
        checks++;
        if (d !== wv) begin
            $display("  [RW] FAIL: INT_ENABLE 写=%h 读=%h", wv, d);
            errors++;
        end
        // DATA_IN: 写随机值读回一致
        wv = $urandom;
        write_reg(8'h18, wv);
        read_reg(8'h18, d);
        checks++;
        if (d !== wv) begin
            $display("  [RW] FAIL: DATA_IN 写=%h 读=%h", wv, d);
            errors++;
        end
        // DATA_OUT: 写随机值读回一致
        wv = $urandom;
        write_reg(8'h1C, wv);
        read_reg(8'h1C, d);
        checks++;
        if (d !== wv) begin
            $display("  [RW] FAIL: DATA_OUT 写=%h 读=%h", wv, d);
            errors++;
        end
        // SCRATCH: 写随机值读回一致
        wv = $urandom;
        write_reg(8'h20, wv);
        read_reg(8'h20, d);
        checks++;
        if (d !== wv) begin
            $display("  [RW] FAIL: SCRATCH 写=%h 读=%h", wv, d);
            errors++;
        end

        $display("== W1S 寄存器（写 1 置位）==");
        // INT_SET: W1S 写 1 置位
        write_reg(8'h28, 32'hFFFF_FFFF);
        read_reg(8'h28, d);
        checks++;
        if (d !== 32'hFFFF_FFFF) begin
            $display("  [W1S] FAIL: INT_SET 写 1 未置位 d=%h", d);
            errors++;
        end

        $display("== RC 寄存器（读清零）==");
        // EVENT_CNT: 复位值 0x000000A5，读后应清零
        read_reg(8'h2C, d);   // 首读
        checks++;
        if (d !== 32'h000000A5) begin
            $display("  [RC] FAIL: EVENT_CNT 首读=%h expect 000000A5", d);
            errors++;
        end
        read_reg(8'h2C, d);   // 再读应为 0
        checks++;
        if (d !== '0) begin
            $display("  [RC] FAIL: EVENT_CNT 读后未清零 d=%h", d);
            errors++;
        end

        $display("== 位域寄存器（逐位属性验证）==");
        // ===== CTRL 位域测试 @0x0C =====
        // CTRL.ENABLE [0:0] RW: 写满该域应可读回
        write_reg(8'h0C, 32'h00000001);
        read_reg(8'h0C, d);
        checks++;
        if ((d & 32'h00000001) !== 32'h00000001) begin
            $display("  [FLD] FAIL: CTRL.ENABLE RW 写满回读 d=%h mask=%h", d, 32'h00000001);
            errors++;
        end
        write_reg(8'h0C, '0);
        // CTRL.MODE [2:1] RW: 写满该域应可读回
        write_reg(8'h0C, 32'h00000006);
        read_reg(8'h0C, d);
        checks++;
        if ((d & 32'h00000006) !== 32'h00000006) begin
            $display("  [FLD] FAIL: CTRL.MODE RW 写满回读 d=%h mask=%h", d, 32'h00000006);
            errors++;
        end
        write_reg(8'h0C, '0);
        // CTRL.IRQ_EN [3:3] W1S: 写 1 置位
        write_reg(8'h0C, '0);   // 先清零（通过 W1C 不可行，直接读初值）
        write_reg(8'h0C, 32'h00000008);   // 写 1 应置位
        read_reg(8'h0C, d);
        checks++;
        if ((d & 32'h00000008) !== 32'h00000008) begin
            $display("  [FLD] FAIL: CTRL.IRQ_EN W1S 未置位 d=%h", d);
            errors++;
        end
        // CTRL.RESERVED [15:4] RSVD: 写忽略、读 0
        write_reg(8'h0C, 32'h0000FFF0);
        read_reg(8'h0C, d);
        checks++;
        if ((d & 32'h0000FFF0) !== '0) begin
            $display("  [FLD] FAIL: CTRL.RESERVED RSVD 非 0 d=%h", d);
            errors++;
        end
        // CTRL.STATUS_CODE [23:16] RC: 读后清零
        do_reset();   // 恢复 RC 位域初值
        read_reg(8'h0C, d);   // 第一次读（应为初值）
        base = d & 32'h00FF0000;
        checks++;
        if (base === '0) begin
            $display("  [FLD] FAIL: CTRL.STATUS_CODE RC 复位后初值为0，读清零不可观测");
            errors++;
        end
        read_reg(8'h0C, d);   // 第二次读应为 0
        checks++;
        if ((d & 32'h00FF0000) !== '0) begin
            $display("  [FLD] FAIL: CTRL.STATUS_CODE RC 读后未清零 d=%h", d);
            errors++;
        end
        // CTRL.FLAGS [31:24] W1C: 写 1 应清零
        do_reset();
        read_reg(8'h0C, d);   // 复位后应有初值
        base = d & 32'hFF000000;
        checks++;
        if (base === '0) begin
            $display("  [FLD] FAIL: CTRL.FLAGS W1C 复位后初值为0，清零不可观测");
            errors++;
        end
        write_reg(8'h0C, 32'hFF000000);
        read_reg(8'h0C, d);
        checks++;
        if ((d & 32'hFF000000) !== '0) begin
            $display("  [FLD] FAIL: CTRL.FLAGS W1C 写 1 未清零 d=%h", d);
            errors++;
        end
        // CTRL 多 RW 域同时写满，验证互不干扰
        write_reg(8'h0C, 32'h00000007);
        read_reg(8'h0C, d);
        checks++;
        if ((d & 32'h00000007) !== 32'h00000007) begin
            $display("  [FLD] FAIL: CTRL 多域写满 d=%h mask=%h", d, 32'h00000007);
            errors++;
        end
        write_reg(8'h0C, '0);

        $display("------------------------------------------------------");
        $display(" regfile-gen result: %0d checks, %0d errors", checks, errors);
        if (errors == 0) begin
            $display("PASS: generated regfile attributes");
            $finish;
        end else begin
            $display("FAIL: %0d errors", errors);
            $fatal(1, "generated regfile test failed");
        end
    end

    initial begin
        $dumpfile("tb_regfile_gen.vcd");
        $dumpvars(0, tb_regfile_gen);
    end

endmodule
`endif // TB_REGFILE_GEN_SV
