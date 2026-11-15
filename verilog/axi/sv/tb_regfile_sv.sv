// ============================================================================
// tb_regfile_sv：验证带属性的 APB 从机（apb_slave_regfile_sv）
// ============================================================================
// 寄存器属性（见 apb_slave_regfile_sv.attr_of）：
//   reg0 (0x00) RO   只读，复位值 0x00010203
//   reg1 (0x04) W1C  写 1 清零
//   reg2 (0x08) RSVD 保留，读 0、写忽略
//   reg3+       RW   读写
//
// 复用 axi_txn_host.execute(txn_t, rsp_t) 驱动事务 —— 该适配器已被主回归
// 234 向量验证。测试里重复实现 AXI 握手时序极易踩 VALID/READY 采样坑，
// 故统一走已验证的抽象层。
// ============================================================================
`ifndef TB_REGFILE_SV_SV
`define TB_REGFILE_SV_SV
`timescale 1ns / 1ps

module tb_regfile_sv #(
    parameter int AW   = 8,
    parameter int DW   = 32,
    parameter int NREG = 16
);

    import bus_txn_pkg::*;
    import axi_defs_pkg::*;

    // 属性寄存器地址
    localparam logic [7:0] A_RO   = 8'h00;
    localparam logic [7:0] A_W1C  = 8'h04;
    localparam logic [7:0] A_RSVD = 8'h08;
    localparam logic [7:0] A_RW   = 8'h0C;

    logic ACLK = 0, ARESETn = 0;
    always #5 ACLK = ~ACLK;

    axi_lite_if #(.AW(AW), .DW(DW)) axi(ACLK);
    apb_if      #(.AW(AW), .DW(DW)) apb(ACLK);

    // 独立协议监视器（被动只读，自动检查 AXI/APB 合法性）
    axi_apb_monitor #(.AW(AW), .DW(DW), .NAME("REGFILE")) mon (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
    );

    axi_apb_bridge_sv #(.AW(AW), .DW(DW)) bridge (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
    );
    apb_slave_regfile_sv #(.AW(AW), .DW(DW), .NREG(NREG), .WAIT_CYCLES(1)) slave (
        .PCLK(ACLK), .PRESETn(ARESETn), .apb(apb)
    );

    // 事务适配器（复用已验证实现）
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
        t.write = wr;
        t.addr  = a;
        t.data  = d;
        return t;
    endfunction

    task automatic write_reg(input [AW-1:0] a, input [DW-1:0] d);
        txn_t t; rsp_t r;
        t = mk(1'b1, a, d);
        host.execute(t, r);
        checks++;
        if (r.resp !== RESP_OKAY) begin
            $display("  [REF] FAIL: 写 addr=%0h resp=%s", a, resp_name(r.resp));
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
            $display("  [REF] FAIL: 读 addr=%0h resp=%s", a, resp_name(r.resp));
            errors++;
        end
    endtask

    // ==================================================================
    task automatic t_ro();
        logic [DW-1:0] d;
        $display("== REF-01: RO 寄存器（只读）==");
        read_reg(A_RO, d);
        checks++;
        if (d !== 32'h0001_0203) begin
            $display("  [REF-01] FAIL: RO 初值=%h expect 00010203", d);
            errors++;
        end
        write_reg(A_RO, 32'hFFFF_FFFF);
        read_reg(A_RO, d);
        checks++;
        if (d !== 32'h0001_0203) begin
            $display("  [REF-01] FAIL: RO 被写改动，值=%h", d);
            errors++;
        end else
            $display("  [REF-01] RO 写被忽略、读回硬件值 00010203 ✓");
    endtask

    task automatic t_w1c();
        logic [DW-1:0] d;
        $display("== REF-02: W1C 寄存器（写 1 清零）==");
        write_reg(A_W1C, 32'hFFFF_FFFF);
        read_reg(A_W1C, d);
        checks++;
        if (d !== 32'h0) begin
            $display("  [REF-02] FAIL: W1C 写全 1 后应 0，实际=%h", d);
            errors++;
        end else
            $display("  [REF-02] W1C 写 1 清零生效 ✓");
        write_reg(A_W1C, 32'h0000_0000);
        read_reg(A_W1C, d);
        checks++;
        if (d !== 32'h0) begin
            $display("  [REF-02] FAIL: W1C 写 0 应保持，实际=%h", d);
            errors++;
        end else
            $display("  [REF-02] W1C 写 0 保持不变 ✓");
    endtask

    task automatic t_rsvd();
        logic [DW-1:0] d;
        $display("== REF-03: RSVD 寄存器（保留）==");
        write_reg(A_RSVD, 32'hDEAD_BEEF);
        read_reg(A_RSVD, d);
        checks++;
        if (d !== 32'h0) begin
            $display("  [REF-03] FAIL: RSVD 读应 0，实际=%h", d);
            errors++;
        end else
            $display("  [REF-03] RSVD 读 0、写忽略 ✓");
    endtask

    task automatic t_rw();
        logic [DW-1:0] d, val;
        $display("== REF-04: RW 寄存器（读写）==");
        for (int k = 0; k < 5; k++) begin
            val = $urandom;
            write_reg(A_RW, val);
            read_reg(A_RW, d);
            checks++;
            if (d !== val) begin
                $display("  [REF-04] FAIL: RW 写=%h 读=%h", val, d);
                errors++;
            end
        end
        $display("  [REF-04] RW 5 次写入读回一致 ✓");
    endtask

    task automatic t_mixed();
        logic [DW-1:0] d;
        $display("== REF-05: 混合地址（属性互不干扰）==");
        write_reg(8'h10, 32'hAAAA_AAAA);
        write_reg(A_RO,  32'h1111_1111);
        write_reg(8'h14, 32'hBBBB_BBBB);
        write_reg(A_RSVD, 32'h2222_2222);
        write_reg(A_W1C, 32'hFFFF_FFFF);

        read_reg(8'h10, d); checks++;
        if (d !== 32'hAAAA_AAAA) begin $display("  [REF-05] RW[0x10]=%h", d); errors++; end
        read_reg(8'h14, d); checks++;
        if (d !== 32'hBBBB_BBBB) begin $display("  [REF-05] RW[0x14]=%h", d); errors++; end
        read_reg(A_RO, d); checks++;
        if (d !== 32'h0001_0203) begin $display("  [REF-05] RO 被影响=%h", d); errors++; end
        read_reg(A_RSVD, d); checks++;
        if (d !== 32'h0) begin $display("  [REF-05] RSVD 被影响=%h", d); errors++; end
        read_reg(A_W1C, d); checks++;
        if (d !== 32'h0) begin $display("  [REF-05] W1C 清零失败=%h", d); errors++; end
        $display("  [REF-05] 5 个寄存器属性互不干扰 ✓");
    endtask

    initial begin
        $display("======================================================");
        $display(" register attributes test (RO/RW/W1C/RSVD)");
        $display("======================================================");
        do_reset();

        t_ro();
        t_w1c();
        t_rsvd();
        t_rw();
        t_mixed();

        $display("------------------------------------------------------");
        if (mon.errors != 0) begin
            $display("  [MON] FAIL: 协议监视器发现 %0d 处违规", mon.errors);
            errors += mon.errors;
        end else
            $display("  [MON] 协议监视器通过（观测 %0d 次 APB 传输）", mon.xfer_count);

        $display(" regfile result: %0d checks, %0d errors", checks, errors);
        if (errors == 0) begin
            $display("PASS: register attributes");
            $finish;
        end else begin
            $display("FAIL: %0d errors", errors);
            $fatal(1, "regfile test failed");
        end
    end

    initial begin
        $dumpfile("tb_regfile_sv.vcd");
        $dumpvars(0, tb_regfile_sv);
    end

endmodule
`endif // TB_REGFILE_SV_SV
