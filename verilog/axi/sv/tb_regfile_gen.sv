// ============================================================================
// tb_regfile_gen：属性验证（由 gen_regfile.py 从 regfile_spec.json 生成）
// ============================================================================
// ⚠️ 自动生成，勿手工编辑。
//
// 覆盖寄存器（10 个）：
    //   0x00  VERSION      RO    reset=0x00010203
    //   0x04  INT_STATUS   W1C   reset=0x00000000
    //   0x08  RESERVED0    RSVD  reset=0x00000000
    //   0x0C  CTRL         RW    reset=0x00000000
    //   0x10  STATUS       RO    reset=0x00000011
    //   0x14  INT_ENABLE   RW    reset=0x00000000
    //   0x18  DATA_IN      RW    reset=0x00000000
    //   0x1C  DATA_OUT     RW    reset=0x00000000
    //   0x20  SCRATCH      RW    reset=0x00000000
    //   0x24  IRQ_CLEAR    W1C   reset=0x00000000
// ============================================================================
`ifndef TB_REGFILE_GEN_SV
`define TB_REGFILE_GEN_SV
`timescale 1ns / 1ps

module tb_regfile_gen #(
    parameter int AW   = 8,
    parameter int DW   = 32,
    parameter int NREG = 10
);

    import bus_txn_pkg::*;
    import axi_defs_pkg::*;

    logic ACLK = 0, ARESETn = 0;
    always #5 ACLK = ~ACLK;

    axi_lite_if #(.AW(AW), .DW(DW)) axi(ACLK);
    apb_if      #(.AW(AW), .DW(DW)) apb(ACLK);

    axi_apb_bridge_sv #(.AW(AW), .DW(DW)) bridge (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
    );
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

    logic [DW-1:0] d, wv;

    initial begin
        $display("======================================================");
        $display(" generated regfile test (10 registers)");
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
        // INT_STATUS: 写全 1 应清零
        write_reg(8'h04, 32'hFFFF_FFFF);
        read_reg(8'h04, d);
        checks++;
        if (d !== '0) begin
            $display("  [W1C] FAIL: INT_STATUS 写 1 后应 0，实际=%h", d);
            errors++;
        end
        // IRQ_CLEAR: 写全 1 应清零
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
        // CTRL: 写随机值读回一致
        wv = $urandom;
        write_reg(8'h0C, wv);
        read_reg(8'h0C, d);
        checks++;
        if (d !== wv) begin
            $display("  [RW] FAIL: CTRL 写=%h 读=%h", wv, d);
            errors++;
        end
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
