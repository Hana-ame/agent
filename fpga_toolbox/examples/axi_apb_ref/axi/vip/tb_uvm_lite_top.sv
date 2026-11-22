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
// 已修复的问题（记录以便参考）
//   曾出现「编译成功但运行无输出」——实际是输出被 stdout 缓冲，
//   程序挂死后缓冲区未 flush，误判为「启动即退出」。
//   真正根因：drive_one/drive_read 在检测到 VALID 后立即撤销 READY，
//   而桥需要 VALID && READY 同拍完成握手，导致桥永远卡在 S_B_RESP。
//   修复：握手判定改为 VALID && READY 同拍，且撤销 READY 前多等一拍。
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
        lite_item it;
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

        // [4] driver 读事务
        t = '0; t.write = 0; t.addr = 8'h10; t.data = '0;
        env.drv.drive_read(t, r);
        checks++;
        if (r.resp === RESP_OKAY && r.data === 32'hA5A5_1234)
            $display("  [4] driver 读回一致 (0x%08h)", r.data);
        else begin
            $display("  [4] FAIL: 读回 data=0x%08h exp=0xA5A51234 resp=%0d",
                     r.data, r.resp);
            errors++;
        end

        // [5] scoreboard 接受正确事务
        it = new(1'b1, 8'h10, 32'h1234_5678, RESP_OKAY);
        env.scb.check_item(it, 32'h1234_5678, RESP_OKAY);
        checks++;
        if (env.scb.compared == 1 && env.scb.mismatches == 0)
            $display("  [5] scoreboard 接受正确事务");
        else begin $display("  [5] FAIL: scb 误判"); errors++; end

        // [6] 反向验证：发现数据错误
        it = new(1'b0, 8'h10, 32'hDEAD_BEEF, RESP_OKAY);
        env.scb.check_item(it, 32'h1234_5678, RESP_OKAY);
        checks++;
        if (env.scb.mismatches == 1)
            $display("  [6] scoreboard 发现数据错误（反向验证）");
        else begin $display("  [6] FAIL: 未发现数据错误"); errors++; end

        // [7] 反向验证：发现响应错误
        it = new(1'b1, 8'h10, 32'h1234, RESP_SLVERR);
        env.scb.check_item(it, 32'h1234, RESP_OKAY);
        checks++;
        if (env.scb.mismatches == 2)
            $display("  [7] scoreboard 发现响应错误（反向验证）");
        else begin $display("  [7] FAIL: 未发现响应错误"); errors++; end

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
