// ============================================================================
// tb_field_iso_sv.sv：CTRL 寄存器跨位域隔离验证
//
// 数据位宽说明（避免误解覆盖范围）：
//   本 TB 的测试向量使用 32 位常量（如 32'hAAAA_BBBB）。
//   在 DW=32 下即覆盖全部数据位；若以 DW=64 编译，
//   高 32 位只会被写 0、不会被本 TB 验证。
//   DW=64 的完整数据通路验证见 tb_dw64_sv（用 64 位向量）。
// ============================================================================
// 背景：
//   寄存器规格中 CTRL 是唯一「单个寄存器内混合 6 种属性」的：
//     ENABLE      [0]      RW
//     MODE        [2:1]    RW
//     IRQ_EN      [3]      W1S   （写 1 置位，写 0 保持）
//     RESERVED    [15:4]   RSVD  （写入忽略，读回 0）
//     STATUS_CODE [23:16]  RC    （读后清零，reset=60）
//     FLAGS       [31:24]  W1C   （写 1 清零，reset=0xA5）
//
//   现有 tb_regfile_gen 逐个验证了各字段行为，但没有验证
//   「写一个域是否会误伤同寄存器的其他域」。这是位域实现最容易出错的地方：
//   掩码写错一位，就会串到相邻域，而逐域测试发现不了。
//
// 本测试验证 8 项：
//   [1] 写 ENABLE 仅影响 bit0
//   [2] 写 MODE 不影响 RESERVED
//   [3] W1S 初始为 0
//   [4] W1S 写 1 置位
//   [5] W1S 写 0 保持原值（关键：W1S 不是 RW）
//   [6] RSVD 写入被忽略
//   [7] W1C 写 1 清零
//   [8] 写 ENABLE 不影响 FLAGS（跨域隔离，最关键的一项）
// ============================================================================
`timescale 1ns/1ps
module tb_field_iso_sv;
  localparam int AW=8,DW=32,NREG=16;
  logic ACLK=0, ARESETn=0;
  always #5 ACLK=~ACLK;
  axi_lite_if #(AW,DW) axi(ACLK);
  apb_if #(AW,DW) apb(ACLK);
  axi_apb_bridge_sv #(AW,DW) bridge(.ACLK(ACLK),.ARESETn(ARESETn),.axi(axi),.apb(apb));
  apb_slave_regfile_gen #(AW,DW) slave(.PCLK(ACLK),.PRESETn(ARESETn),.apb(apb));
  axi_txn_host #(AW,DW) host(.ACLK(ACLK),.ARESETn(ARESETn),.axi(axi));

  import bus_txn_pkg::*; import axi_defs_pkg::*;
  int errors=0, checks=0;
  localparam CTRL=8'h0C;

  task automatic wr(input [7:0] a, input [31:0] d, output bit [1:0] r);
    txn_t t; rsp_t rp; t='0; t.write=1; t.addr=a; t.data=d; host.execute(t,rp); r=rp.resp;
  endtask
  task automatic rd(input [7:0] a, output [31:0] d, output bit [1:0] r);
    txn_t t; rsp_t rp; t='0; t.write=0; t.addr=a; host.execute(t,rp); d=rp.data; r=rp.resp;
  endtask

  initial begin
    bit [1:0] r; bit [31:0] d;
    $display("=== CTRL 跨位域隔离验证 ===");
    axi.AWVALID=0; axi.WVALID=0; axi.ARVALID=0; axi.BREADY=0; axi.RREADY=0;
    ARESETn=0; repeat(5) @(posedge ACLK); ARESETn=1; @(posedge ACLK);

    // [1] 写 ENABLE（bit0）不应影响其他域
    wr(CTRL, 32'h0000_0001, r);
    rd(CTRL, d, r);
    checks++;
    if (d[0]===1'b1 && d[2:1]===2'b00 && d[3]===1'b0)
      $display("  ✓ 写 ENABLE 仅影响 bit0（d=0x%08h）", d);
    else begin $display("  ✗ ENABLE 污染其他域 d=0x%08h", d); errors++; end

    // [2] 写 MODE（bits2:1）
    wr(CTRL, 32'h0000_0006, r);   // MODE=11
    rd(CTRL, d, r);
    checks++;
    if (d[2:1]===2'b11 && d[15:4]===12'h0)
      $display("  ✓ 写 MODE 正确且不影响 RESERVED");
    else begin $display("  ✗ MODE 异常 d=0x%08h", d); errors++; end

    // [3] IRQ_EN 是 W1S：写 1 置位，写 0 不变
    wr(CTRL, 32'h0000_0000, r);   // 全写 0
    rd(CTRL, d, r);
    checks++;
    if (d[3]===1'b0) $display("  ✓ W1S 初始为 0");
    else begin $display("  ✗ W1S 初值异常 d=0x%08h", d); errors++; end

    wr(CTRL, 32'h0000_0008, r);   // 写 1 到 IRQ_EN
    rd(CTRL, d, r);
    checks++;
    if (d[3]===1'b1) $display("  ✓ W1S 写 1 置位");
    else begin $display("  ✗ W1S 未置位 d=0x%08h", d); errors++; end

    wr(CTRL, 32'h0000_0000, r);   // 再写 0
    rd(CTRL, d, r);
    checks++;
    if (d[3]===1'b1) $display("  ✓ W1S 写 0 保持原值（不被清零）");
    else begin $display("  ✗ W1S 被写 0 清零 d=0x%08h", d); errors++; end

    // [4] RESERVED 是 RSVD：写入无效，读回 0
    wr(CTRL, 32'h0000_FFF0, r);   // 写满 RESERVED
    rd(CTRL, d, r);
    checks++;
    if (d[15:4]===12'h0) $display("  ✓ RSVD 写入被忽略（读回 0）");
    else begin $display("  ✗ RSVD 竟然可写 d=0x%08h", d); errors++; end

    // [5] FLAGS 是 W1C：写 1 清零
    wr(CTRL, 32'hFF00_0000, r);   // 0xFF = 全 1 → 应清零
    rd(CTRL, d, r);
    checks++;
    if (d[31:24]===8'h00) $display("  ✓ W1C 写 1 清零");
    else begin $display("  ✗ W1C 未清零 d=0x%08h", d); errors++; end

    // [6] 关键：写一个域不应误触发其他属性
    //   写 MODE（RW）时 RESERVED 位为 0 → 不应影响 FLAGS
    wr(CTRL, 32'h0000_0000, r);
    rd(CTRL, d, r);   // 先读一次，可能已清 STATUS_CODE
    begin
      bit [7:0] flags_before; bit [7:0] flags_after;
      // 设置 FLAGS 为已知值（W1C 只能清零，先读 reset 值 0xA5）
      ARESETn=0; repeat(3) @(posedge ACLK); ARESETn=1; @(posedge ACLK);
      rd(CTRL, d, r);
      flags_before = d[31:24];
      // 只写 ENABLE，不动 FLAGS
      wr(CTRL, 32'h0000_0001, r);
      rd(CTRL, d, r);
      flags_after = d[31:24];
      checks++;
      if (flags_after === flags_before)
        $display("  ✓ 写 ENABLE 未影响 FLAGS（0x%02h→0x%02h）", flags_before, flags_after);
      else begin
        $display("  ✗ FLAGS 被误改 0x%02h→0x%02h", flags_before, flags_after);
        errors++;
      end
    end

    $display("------------------------------------------------------");
    $display(" field-iso result: %0d checks, %0d errors", checks, errors);
    if (errors==0) begin $display("PASS: CTRL cross-field isolation"); $finish; end
    else $fatal(1);
  end
endmodule
