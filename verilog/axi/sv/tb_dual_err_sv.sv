// ============================================================================
// tb_dual_err_sv.sv：并发桥错误响应隔离验证
// ============================================================================
// 背景：
//   并发桥（axi_apb_bridge_dual_sv）用两个独立标志传递错误：
//     w_slverr —— 写通道的 APB 错误
//     r_slverr —— 读通道的 APB 错误
//   但现有 tb_dual_sv / tb_dual_stress 完全没有测 exp_err，
//   即并发桥的错误路径此前从未被验证过。
//
// 本测试验证 8 项：
//   [1] 越界写 → SLVERR
//   [2] 错误后正常写 → OKAY（错误标志不残留）
//   [3] 越界读 → SLVERR
//   [4] 错误后正常读 → OKAY 且数据正确
//   [5] 写错误置位 w_slverr
//   [6] 写错误不影响后续读（r_slverr 独立）
//   [7] 读错误置位 r_slverr
//   [8] 读错误不影响后续写（w_slverr 独立）
//
// [6][8] 是关键：并发桥读写并行，若两个错误标志串了，
// 会出现「写错导致读报错」这类交叉污染。
// ============================================================================
`timescale 1ns/1ps
module tb_dual_err_sv;
  localparam int AW=8,DW=32,NREG=16;
  logic ACLK=0, ARESETn=0;
  always #5 ACLK=~ACLK;
  axi_lite_if #(AW,DW) axi(ACLK);
  apb_if #(AW,DW) apb(ACLK);
  axi_apb_bridge_dual_sv #(AW,DW) bridge(.ACLK(ACLK),.ARESETn(ARESETn),.axi(axi),.apb(apb));
  apb_slave_regs_sv #(AW,DW,NREG,1) slave(.PCLK(ACLK),.PRESETn(ARESETn),.apb(apb));
  axi_txn_host #(AW,DW) host(.ACLK(ACLK),.ARESETn(ARESETn),.axi(axi));

  import bus_txn_pkg::*; import axi_defs_pkg::*;
  int errors=0, checks=0;

  task automatic wr(input [7:0] a, input [31:0] d, output bit [1:0] r);
    txn_t t; rsp_t rp; t='0; t.write=1; t.addr=a; t.data=d; host.execute(t,rp); r=rp.resp;
  endtask
  task automatic rd(input [7:0] a, output [31:0] d, output bit [1:0] r);
    txn_t t; rsp_t rp; t='0; t.write=0; t.addr=a; host.execute(t,rp); d=rp.data; r=rp.resp;
  endtask

  initial begin
    bit [1:0] r; bit [31:0] d;
    $display("======================================================");
    $display(" 并发桥错误响应隔离验证");
    $display("======================================================");
    axi.AWVALID=0; axi.WVALID=0; axi.ARVALID=0; axi.BREADY=0; axi.RREADY=0;
    ARESETn=0; repeat(5) @(posedge ACLK); ARESETn=1; @(posedge ACLK);

    checks++; wr(8'hFF, 32'h1, r);
    if (r === RESP_SLVERR) $display("  ✓ 越界写 → SLVERR");
    else begin $display("  ✗ 越界写 resp=%0d", r); errors++; end

    checks++; wr(8'h10, 32'hAAAA, r);
    if (r === RESP_OKAY) $display("  ✓ 错误后正常写 → OKAY");
    else begin $display("  ✗ 正常写 resp=%0d（错误标志残留）", r); errors++; end

    checks++; rd(8'hFF, d, r);
    if (r === RESP_SLVERR) $display("  ✓ 越界读 → SLVERR");
    else begin $display("  ✗ 越界读 resp=%0d", r); errors++; end

    checks++; rd(8'h10, d, r);
    if (r === RESP_OKAY && d === 32'hAAAA) $display("  ✓ 错误后正常读 → OKAY/0x%h", d);
    else begin $display("  ✗ 正常读 data=%h resp=%0d", d, r); errors++; end

    // 关键：读写错误标志是否独立（w_slverr / r_slverr 分开）
    checks++; wr(8'hFF, 32'h2, r);
    if (r === RESP_SLVERR) $display("  ✓ 写错误置 w_slverr");
    else errors++;
    checks++; rd(8'h10, d, r);
    if (r === RESP_OKAY) $display("  ✓ 写错误不影响后续读（r_slverr 独立）");
    else begin $display("  ✗ 读被写错误污染 resp=%0d", r); errors++; end

    checks++; rd(8'hFF, d, r);
    if (r === RESP_SLVERR) $display("  ✓ 读错误置 r_slverr");
    else errors++;
    checks++; wr(8'h10, 32'hBBBB, r);
    if (r === RESP_OKAY) $display("  ✓ 读错误不影响后续写（w_slverr 独立）");
    else begin $display("  ✗ 写被读错误污染 resp=%0d", r); errors++; end

    // ---- 并发场景：读写同时出错时互不污染 ----
    $display("== 并发场景：读写同时发起 ==");
    begin
        // 都越界：都应 SLVERR
        begin
            txn_t tw, tr; rsp_t rw, rr;
            tw='0; tw.write=1; tw.addr=8'hFF; tw.data=32'h1;
            tr='0; tr.write=0; tr.addr=8'hFE;
            fork
              begin host.execute(tw, rw); end
              begin host.execute(tr, rr); end
            join
            checks++;
            if (rw.resp === RESP_SLVERR && rr.resp === RESP_SLVERR)
                $display("  ✓ 并发双越界：写/读均 SLVERR");
            else begin
                $display("  ✗ 并发双越界：写=%0d 读=%0d", rw.resp, rr.resp);
                errors++;
            end
        end
        // 写错读对
        begin
            txn_t tw, tr; rsp_t rw, rr;
            tw='0; tw.write=1; tw.addr=8'hFF; tw.data=32'h2;
            tr='0; tr.write=0; tr.addr=8'h10;
            fork
              begin host.execute(tw, rw); end
              begin host.execute(tr, rr); end
            join
            checks++;
            if (rw.resp === RESP_SLVERR && rr.resp === RESP_OKAY)
                $display("  ✓ 并发混合：写 SLVERR / 读 OKAY（互不污染）");
            else begin
                $display("  ✗ 并发混合：写=%0d 读=%0d", rw.resp, rr.resp);
                errors++;
            end
        end
        // 写对读错
        begin
            txn_t tw, tr; rsp_t rw, rr;
            tw='0; tw.write=1; tw.addr=8'h10; tw.data=32'hABCD;
            tr='0; tr.write=0; tr.addr=8'hFD;
            fork
              begin host.execute(tw, rw); end
              begin host.execute(tr, rr); end
            join
            checks++;
            if (rw.resp === RESP_OKAY && rr.resp === RESP_SLVERR)
                $display("  ✓ 并发混合2：写 OKAY / 读 SLVERR（互不污染）");
            else begin
                $display("  ✗ 并发混合2：写=%0d 读=%0d", rw.resp, rr.resp);
                errors++;
            end
            rd(8'h10, d, r);
            checks++;
            if (r === RESP_OKAY && d === 32'hABCD)
                $display("  ✓ 并发后数据一致：0x%h", d);
            else begin
                $display("  ✗ 数据=%h resp=%0d", d, r);
                errors++;
            end
        end
    end

    $display("------------------------------------------------------");
    $display(" dual-err result: %0d checks, %0d errors", checks, errors);
    if (errors==0) begin $display("PASS: dual bridge error isolation"); $finish; end
    else begin $display("FAIL"); $fatal(1); end
  end
endmodule
