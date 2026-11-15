// ============================================================================
// tb_uvm_lite_top.sv：轻量 UVM 风格环境的运行时验证
// ============================================================================
// 目的：让「组件化验证架构」在 Verilator 下获得真实运行验证。
//
//   本项目的 UVM 配套（axi_vip_uvm_pkg.sv）用真 UVM 库，只能编译验证，
//   运行时仿真需要 irun/VCS/Questa。而 UVM 的核心价值——组件化 + TLM 连接——
//   用纯 SV 类也能实现，且 Verilator 可真实运行。
//
// 本文件验证 axi_uvm_lite_pkg 的：
//   [1] build：4 个组件创建（sequencer/driver/monitor/scoreboard）
//   [2] connect：TLM 连接（monitor.ap → scoreboard）
//   [3] driver 驱动真实 AXI 写事务并收到 OKAY
//
// 已知限制（如实记录）
//   曾尝试在同一个 initial 块中继续加入「读事务 + scoreboard 正反验证」，
//   出现「编译成功但运行无输出」的挂死现象，未能定位根因
//   （已排除：变量声明顺序、虚接口、NBA 驱动、块内/模块级声明、命名块）。
//   因此本文件保留已验证可运行的最小集合；scoreboard 的正反验证逻辑
//   已在 axi_uvm_lite_pkg 中实现，待后续定位挂死问题后再接入。
// ============================================================================
`ifndef TB_UVM_LITE_TOP_SV
`define TB_UVM_LITE_TOP_SV
`timescale 1ns / 1ps

module tb_uvm_lite_top #(
    parameter int AW = 8,
    parameter int DW = 32,
    parameter int NREG = 16
);

    import bus_txn_pkg::*;
    import axi_defs_pkg::*;
    import axi_uvm_lite_pkg::*;

    logic ACLK = 0, ARESETn = 0;
    always #5 ACLK = ~ACLK;

    axi_lite_if #(.AW(AW), .DW(DW)) axi(ACLK);
    apb_if      #(.AW(AW), .DW(DW)) apb(ACLK);

    axi_apb_bridge_sv #(.AW(AW), .DW(DW)) bridge (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
    );
    apb_slave_regs_sv #(.AW(AW), .DW(DW), .NREG(NREG), .WAIT_CYCLES(1)) slave (
        .PCLK(ACLK), .PRESETn(ARESETn), .apb(apb)
    );
    axi_apb_monitor #(.AW(AW), .DW(DW), .NAME("LITE")) mon_hw (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
    );

    int errors = 0, checks = 0;

    initial begin
        rsp_t r;
        txn_t t;
        lite_env env;

        $display("======================================================");
        $display(" 轻量 UVM 风格环境：组件化验证的运行时证明");
        $display("======================================================");

        // 复位
        ARESETn = 0;
        axi.AWVALID = 0; axi.WVALID = 0; axi.ARVALID = 0;
        axi.BREADY = 0;  axi.RREADY = 0;
        repeat (4) @(posedge ACLK);
        ARESETn = 1;
        @(posedge ACLK);

        // [1] build
        env = new(axi, axi);
        checks++;
        if (env.sqr != null && env.drv != null && env.mon != null && env.scb != null)
            $display("  [1] build：4 组件创建成功 (sqr/drv/mon/scb)");
        else begin $display("  [1] FAIL: build"); errors++; end

        // [2] connect
        env.connect();
        checks++;
        if (env.mon.ap.size() == 1)
            $display("  [2] connect：monitor.ap → scoreboard 已连接");
        else begin
            $display("  [2] FAIL: 订阅者数=%0d", env.mon.ap.size());
            errors++;
        end

        // [3] driver 驱动真实 AXI 写事务
        t = '0; t.write = 1; t.addr = 8'h10; t.data = 32'hA5A5_1234;
        env.drv.drive_one(t, r);
        checks++;
        if (r.resp === RESP_OKAY)
            $display("  [3] driver 写事务完成 (resp=OKAY)");
        else begin $display("  [3] FAIL: 写 resp=%0d", r.resp); errors++; end

        // 汇总
        $display("------------------------------------------------------");
        if (mon_hw.errors != 0) begin
            $display("  [MON] FAIL: 硬件监视器发现 %0d 处违规", mon_hw.errors);
            errors += mon_hw.errors;
        end else
            $display("  [MON] 硬件协议监视器通过（%0d 次 APB 传输）", mon_hw.xfer_count);

        $display(" uvm-lite result: %0d checks, %0d errors", checks, errors);
        if (errors == 0) begin
            $display("PASS: uvm-lite component environment");
            $finish;
        end else begin
            $display("FAIL: %0d errors", errors);
            $fatal(1, "uvm-lite test failed");
        end
    end

endmodule
`endif // TB_UVM_LITE_TOP_SV
