// ============================================================================
// axi_lite_vip：AXI4-Lite 验证 IP（单模块，直接接入任意 DUT）
// ============================================================================
// 一个模块内包含 打激励 + 采集 + 验证激励，通过 interface 接入 DUT：
//
//   ┌──────────────────────── axi_lite_vip ────────────────────────┐
//   │                                                               │
//   │  sequence ──▶ driver ──▶ [axi] ──▶ DUT                        │
//   │                  │                  │                        │
//   │                  └── ref_model ◀────┘                        │
//   │                          │                                    │
//   │                      scoreboard ──▶ stats ──▶ report()        │
//   │                                                               │
//   └───────────────────────────────────────────────────────────────┘
//
// 接入（三行）：
//     axi_lite_vip #(.AW(8),.DW(32)) vip(.ACLK(clk), .ARESETn(rst_n), .axi(axi));
//     vip.setup(.limit(64), .verbose(0));
//     vip.push_back(seq_vec);   // 或 vip.run_sequence(seq);
//     vip.report();
//
// 注意：VIP 占用 AXI master 角色，DUT 必须接在 slave 侧。
// ============================================================================
`ifndef AXI_LITE_VIP_SV
`define AXI_LITE_VIP_SV
`timescale 1ns / 1ps

module axi_lite_vip #(
    parameter int AW = 8,
    parameter int DW = 32
)(
    input  logic ACLK,
    input  logic ARESETn,
    axi_lite_if.master axi      // VIP 是 master，驱动到 DUT
);

    import bus_txn_pkg::*;
    import axi_defs_pkg::*;
    import axi_vip_pkg::*;

    // ---- VIP 内部状态 ----
    axi_vip_stats  stats;
    axi_ref_model  model;
    txn_t          queue[$];          // 待执行事务队列
    bit [31:0]     limit_addr = 64;   // 越界阈值（默认 16 reg * 4B）
    bit            verbose    = 0;

    initial begin
        stats = new();
        model = new();
        model.has_mem = 1;            // VIP 自带寄存镜像（DUT 为寄存器堆时启用）
        model.word_addressed = 1;     // 默认按字(addr>>2)寻址，与 apb_slave_regs 对齐
    end

    // ==================================================================
    // 对外 API：配置
    // ==================================================================
    //   limit   : 越界阈值（地址 >= limit 期望 SLVERR）
    //   vb      : 详细日志
    //   word_ma : 1=按字(addr>>2)寻址（寄存器堆 DUT），0=按字节寻址
    task automatic setup(input bit [31:0] limit, input bit vb = 0,
                         input bit word_ma = 1);
        limit_addr = limit;
        verbose    = vb;
        model.word_addressed = word_ma;
    endtask

    // 压入一笔事务
    function automatic void push_back(input txn_t t);
        queue.push_back(t);
    endfunction

    // 压入一串（sequence 对象）
    function automatic void run_sequence(input axi_sequence seq);
        foreach (seq.q[i]) queue.push_back(seq.q[i]);
    endfunction

    function automatic int pending();
        return queue.size();
    endfunction

    // ==================================================================
    // ② driver：把 txn_t 展开成 AXI 握手时序
    // ==================================================================
    task automatic drive(input txn_t t);
        if (t.write) begin
            if (t.same) begin
                // AW + W 同拍
                @(posedge ACLK);
                axi.AWVALID <= 1'b1; axi.AWADDR <= t.addr;
                axi.WVALID  <= 1'b1; axi.WDATA  <= t.data;
                do @(posedge ACLK); while (!(axi.AWVALID && axi.AWREADY));
                axi.AWVALID <= 1'b0;
                do @(posedge ACLK); while (!(axi.WVALID && axi.WREADY));
                axi.WVALID  <= 1'b0;
            end else begin
                // AW 先，W 后
                @(posedge ACLK);
                axi.AWVALID <= 1'b1; axi.AWADDR <= t.addr;
                do @(posedge ACLK); while (!(axi.AWVALID && axi.AWREADY));
                axi.AWVALID <= 1'b0;

                @(posedge ACLK);
                axi.WVALID <= 1'b1; axi.WDATA <= t.data;
                do @(posedge ACLK); while (!(axi.WVALID && axi.WREADY));
                axi.WVALID <= 1'b0;
            end
        end else begin
            @(posedge ACLK);
            axi.ARVALID <= 1'b1; axi.ARADDR <= t.addr;
            do @(posedge ACLK); while (!(axi.ARVALID && axi.ARREADY));
            axi.ARVALID <= 1'b0;
        end

        // pause：拉低全部 VALID，插 N 拍空闲
        if (t.pause != 0) begin
            axi.AWVALID <= 1'b0;
            axi.WVALID  <= 1'b0;
            axi.ARVALID <= 1'b0;
            repeat (t.pause) @(posedge ACLK);
        end
    endtask

    // ==================================================================
    // ③ monitor：采集响应（写 B 或读 R），并与参考模型比对
    // ==================================================================
    task automatic collect_write(input txn_t t);
        rsp_t exp_r, act_r;
        exp_r = model.predict(t, limit_addr);

        axi.BREADY <= 1'b1;
        do @(posedge ACLK); while (!axi.BVALID);
        act_r = exp_r;
        act_r.resp = axi.BRESP;
        axi.BREADY <= 1'b0;
        @(posedge ACLK);

        check(t, exp_r, act_r);
    endtask

    task automatic collect_read(input txn_t t);
        rsp_t exp_r, act_r;
        exp_r = model.predict(t, limit_addr);

        axi.RREADY <= 1'b1;
        do @(posedge ACLK); while (!axi.RVALID);
        act_r = exp_r;
        act_r.data = axi.RDATA;
        act_r.resp = axi.RRESP;
        axi.RREADY <= 1'b0;
        @(posedge ACLK);

        check(t, exp_r, act_r);
    endtask

    // ==================================================================
    // ④ scoreboard：验证激励——期望 vs 实际
    // ==================================================================
    task automatic check(input txn_t t, input rsp_t exp_r, input rsp_t act_r);
        stats.checks++;
        stats.txns++;
        if (t.write) stats.writes++; else stats.reads++;
        stats.pause_total += t.pause;
        if (act_r.resp == RESP_SLVERR) stats.slverr_cnt++;

        // 1) 响应检查：t.exp_err → 期望 SLVERR；否则期望 OKAY
        if (t.exp_err) begin
            if (act_r.resp !== RESP_SLVERR) begin
                $display("VIP FAIL[%0d]: %s addr=%0h expect SLVERR got %s",
                         stats.checks, t.write ? "WR" : "RD", t.addr,
                         resp_name(act_r.resp));
                stats.errors++;
            end
        end else begin
            if (act_r.resp !== RESP_OKAY) begin
                $display("VIP FAIL[%0d]: %s addr=%0h expect OKAY got %s",
                         stats.checks, t.write ? "WR" : "RD", t.addr,
                         resp_name(act_r.resp));
                stats.errors++;
            end else if (!t.write && act_r.data !== exp_r.data) begin
                $display("VIP FAIL[%0d]: RD addr=%0h exp=%h got=%h",
                         stats.checks, t.addr, exp_r.data, act_r.data);
                stats.errors++;
            end
        end

        if (verbose)
            $display("  VIP: %s addr=%0h data=%h resp=%s",
                     t.write ? "WR" : "RD", t.addr, t.data, resp_name(act_r.resp));
    endtask

    // ==================================================================
    // 主执行体：顺序消费队列
    // ==================================================================
    task automatic execute_all();
        foreach (queue[i]) begin
            drive(queue[i]);
            if (queue[i].write) collect_write(queue[i]);
            else                collect_read(queue[i]);
        end
        queue.delete();
    endtask

    // ==================================================================
    // ⑤ report：结果汇总（对应 UVM report server）
    // ==================================================================
    function automatic bit report();
        stats.dump("VIP");
        if (stats.errors == 0) begin
            $display("VIP PASS: %0d checks, 0 errors", stats.checks);
            return 1'b1;
        end else begin
            $display("VIP FAIL: %0d errors in %0d checks", stats.errors, stats.checks);
            return 1'b0;
        end
    endfunction

endmodule
`endif // AXI_LITE_VIP_SV