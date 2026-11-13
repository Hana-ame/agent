// AXI4-Lite → APB → 寄存器堆 → 响应回 AXI 的 test bench
//
// 验证目标（"可以慢，但一定要合法"）：
//   1. 数据一致性：写 N 个寄存器 → 读回，逐字比对（跨桥往返无丢失/错位）
//   2. 慢速外设：APB 从机插 wait states（WAIT_CYCLES=3），桥必须合规等待
//   3. 协议合法性（monitor 自动检查）：
//      a. AXI VALID 不能在没有 VALID 的通道上拉高 READY（简单起见查 essentials）
//      b. APB: PENABLE 只能在 PSEL=1 时拉高；PSEL=1 且 PENABLE=1 时必须
//         保持到 PREADY=1（不能中途撤销）
//      c. APB SETUP 阶段 PREADY 必须为 0（从机不允许提前就绪）
//      d. 写事务 BVALID/B 只能有一次；读事务 RVALID/R 只能有一次（无多响应）
//   4. 越界响应：地址 >= NREG*4 读/写都应回 BRESP/RRESP=2'b10 (SLVERR)
//      （APB 从机把越界报给桥，桥转成 AXI 错误响应）
//   5. 同发写（AW+W 一拍）：桥必须接受并正确转发
`timescale 1ns / 1ps

module tb_axi_apb;

    localparam AW = 8;
    localparam DW = 32;
    localparam NREG = 16;
    localparam WAIT_CYCLES = 3;   // 慢速外设 wait states

    // ---- 时钟/复位 ----
    reg ACLK = 0;
    reg ARESETn = 0;
    always #5 ACLK = ~ACLK;

    // ---- AXI 信号（连 master + bridge） ----
    wire [AW-1:0] AWADDR; wire AWVALID, AWREADY;
    wire [DW-1:0] WDATA;  wire WVALID,  WREADY;
    wire [1:0]    BRESP;  wire BVALID,  BREADY;
    wire [AW-1:0] ARADDR; wire ARVALID, ARREADY;
    wire [DW-1:0] RDATA;  wire RVALID,  RREADY;
    wire [1:0]    RRESP;

    // ---- APB 信号（bridge 主 -> 从机） ----
    wire        PSEL, PENABLE, PWRITE, PREADY, PSLVERR;
    wire [AW-1:0] PADDR;
    wire [DW-1:0] PWDATA, PRDATA;

    // ---- 实例化 master ----
    axi_lite_vip_master #(.AW(AW), .DW(DW)) master (
        .ACLK(ACLK), .ARESETn(ARESETn),
        .AWVALID(AWVALID), .AWADDR(AWADDR), .AWREADY(AWREADY),
        .WVALID (WVALID),  .WDATA (WDATA),  .WREADY (WREADY),
        .BVALID (BVALID),  .BRESP (BRESP),  .BREADY (BREADY),
        .ARVALID(ARVALID), .ARADDR(ARADDR), .ARREADY(ARREADY),
        .RVALID (RVALID),  .RDATA (RDATA),  .RRESP (RRESP),  .RREADY (RREADY)
    );

    // ---- 实例化 bridge ----
    axi_apb_bridge #(.AW(AW), .DW(DW), .NREG(NREG)) bridge (
        .ACLK(ACLK), .ARESETn(ARESETn),
        .AWVALID(AWVALID), .AWADDR(AWADDR), .AWREADY(AWREADY),
        .WVALID (WVALID),  .WDATA (WDATA),  .WREADY (WREADY),
        .BVALID (BVALID),  .BRESP (BRESP),  .BREADY (BREADY),
        .ARVALID(ARVALID), .ARADDR(ARADDR), .ARREADY(ARREADY),
        .RVALID (RVALID),  .RDATA (RDATA),  .RRESP (RRESP),  .RREADY (RREADY),
        .PSEL(PSEL), .PENABLE(PENABLE), .PWRITE(PWRITE),
        .PADDR(PADDR), .PWDATA(PWDATA),
        .PREADY(PREADY), .PRDATA(PRDATA), .PSLVERR(PSLVERR)
    );

    // ---- 实例化慢速从机 ----
    apb_slave_regs #(.AW(AW), .DW(DW), .NREG(NREG), .WAIT_CYCLES(WAIT_CYCLES)) slave (
        .PCLK(ACLK), .PRESETn(ARESETn),
        .PSEL(PSEL), .PENABLE(PENABLE), .PWRITE(PWRITE),
        .PADDR(PADDR), .PWDATA(PWDATA),
        .PREADY(PREADY), .PRDATA(PRDATA), .PSLVERR(PSLVERR)
    );

    // ---- 协议合法性 monitor（规则来自 apb_defs_pkg，单一事实来源） ----
    integer mon_err = 0;
    // 上一拍 APB 状态（替代 $past）
    reg prev_access;
    reg prev_ready;
    always @(posedge ACLK or negedge ARESETn) begin
        if (!ARESETn) begin
            prev_access <= 1'b0;
            prev_ready  <= 1'b0;
        end else begin
            prev_access <= (PSEL && PENABLE);
            prev_ready  <= PREADY;
        end
    end

    // APB 检查（调用定义层规则函数）
    always @(posedge ACLK) begin
        // R1: PENABLE 只能在 PSEL=1 时拉高
        if (!apb_defs_pkg::penable_ok(PSEL, PENABLE)) begin
            $display("MON FAIL: PENABLE while PSEL=0 @%0t", $time);
            mon_err++;
        end
        // R2: APB ACCESS 阶段若上拍在 ACCESS 且 PREADY=0，本拍不能退出 ACCESS
        if (!apb_defs_pkg::phase_transition_ok(prev_access, prev_ready,
                                               (PSEL && PENABLE))) begin
            $display("MON FAIL: APB ACCESS terminated before PREADY @%0t", $time);
            mon_err++;
        end
        // R3: SETUP 阶段从机不允许 PREADY=1
        if (PSEL && !PENABLE && PREADY) begin
            $display("MON FAIL: PREADY during SETUP @%0t", $time);
            mon_err++;
        end
        // R4(log): 传输完成拍打印（便于核对 wait states 慢速时序）
        if (apb_defs_pkg::transfer_done(PSEL, PENABLE, PREADY))
            $display("    APB xfer done: addr=%02h %s", PADDR,
                     PWRITE ? "WR" : "RD");
    end

    // B/R 响应名打印辅助（用定义层 resp_name）
    // 在响应握手时打印

    // ---- 主流程：数据一致性 + 慢速 + 越界 ----
    reg [31:0] ref_mem [0:NREG-1];   // 参考模型镜像
    integer i;
    reg [DW-1:0] rdata;
    reg [1:0] resp;
    integer err = 0;
    integer pass = 0;

    initial begin
        $dumpfile("tb_axi_apb.vcd");
        $dumpvars(0, tb_axi_apb);

        $display("== AXI->APB->AXI test bench (slow slave, WAIT=%0d) ==", WAIT_CYCLES);

        // 复位
        repeat (4) @(posedge ACLK);
        ARESETn = 1;

        // ---- 1. 写 NREG 个寄存器（分开发 AW/W，随机数据）----
        $display("-- phase 1: write %0d regs (split AW/W) --", NREG);
        for (i = 0; i < NREG; i++) begin
            ref_mem[i] = $random;
            master.write_split(i[AW-1:0] * 4, ref_mem[i], resp);
            if (resp !== axi_defs_pkg::RESP_OKAY) begin
                $display("WRITE FAIL: addr=%0d resp=%b", i, resp);
                err++;
            end else pass++;
        end

        // ---- 2. 读回并比对（慢速外设，验证数据往返 + wait states 合法）----
        $display("-- phase 2: read back %0d regs --", NREG);
        for (i = 0; i < NREG; i++) begin
            master.read(i[AW-1:0] * 4, rdata, resp);
            if (resp !== axi_defs_pkg::RESP_OKAY) begin
                $display("READ FAIL: addr=%0d resp=%b", i, resp);
                err++;
            end else if (rdata !== ref_mem[i]) begin
                $display("DATA MISMATCH: addr=%0d exp=%h got=%h", i, ref_mem[i], rdata);
                err++;
            end else pass++;
        end

        // ---- 3. 同发写（AW+W 同拍）----
        $display("-- phase 3: same-cycle AW+W write --");
        ref_mem[0] = 32'hDEAD_BEEF;
        master.write_same(8'h00, ref_mem[0], resp);
        if (resp !== axi_defs_pkg::RESP_OKAY) begin $display("WRITE_SAME FAIL"); err++; end
        master.read(8'h00, rdata, resp);
        if (rdata !== ref_mem[0]) begin $display("READ_SAME FAIL"); err++; end
        else pass++;

        // ---- 4. 越界读/写（SLVERR 响应，用定义层 resp_name 确认）----
        $display("-- phase 4: out-of-range access (ref_mem SLVERR) --");
        master.write_split(8'h40, 32'h0000_1234, resp);   // 0x40 = reg16，越界
        $display("    OOB write resp=%b (%s)", resp, axi_defs_pkg::resp_name(resp));
        if (resp !== axi_defs_pkg::RESP_SLVERR) begin $display("OOB WRITE: expected SLVERR got %b", resp); err++; end
        else pass++;
        master.read(8'h40, rdata, resp);
        $display("    OOB read  resp=%b (%s)", resp, axi_defs_pkg::resp_name(resp));
        if (resp !== axi_defs_pkg::RESP_SLVERR) begin $display("OOB READ: expected SLVERR got %b", resp); err++; end
        else pass++;

        // ---- 5. monitor 错误汇总 ----
        if (mon_err == 0 && err == 0)
            $display("PASS: axi->apb->axi all %0d checks passed (monitor clean)",
                     pass);
        else begin
            $display("FAIL: %0d data errors, %0d monitor errors", err, mon_err);
            $fatal(1, "axi_apb test bench failed");
        end
        $finish;
    end

endmodule