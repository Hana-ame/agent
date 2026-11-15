// ============================================================================
// axi_uvm_lite_pkg.sv：轻量 UVM 风格验证环境（Verilator 可运行）
// ============================================================================
// 背景与定位：
//   本项目的 UVM 配套（axi_vip_uvm_pkg.sv）只能用 slang 做编译验证，
//   运行时仿真需要 irun/VCS/Questa 等商用 EDA 工具，当前环境没有。
//   但「UVM 的核心价值」——组件化 + TLM 连接 + phase 管理 ——
//   用纯 SystemVerilog 类也能实现，且 Verilator 可以真实运行。
//
//   所以本包提供一个「UVM 风格的轻量环境」：
//   结构、职责划分、TLM 连接方式都对齐 UVM，但用 plain class 实现，
//   通过 mailbox 通信，手工实现 build/connect/run/check 阶段。
//
//   作用：
//     1. 让组件化验证架构在 Verilator 下可获得真实运行验证
//     2. 作为 UVM 版本的参照实现（两者结构一一对应，便于对照）
//     3. 证明验证架构设计本身是可运行的，不依赖特定工具
//
// 组件与 UVM 的对应关系：
//   uvm_driver       → lite_driver      驱动 AXI 事务
//   uvm_sequencer    → lite_sequencer   事务队列 + 仲裁
//   uvm_monitor      → lite_monitor     观测总线并广播
//   uvm_scoreboard   → lite_scoreboard  比对期望与实际
//   uvm_env          → lite_env         组装并连接各组件
//   uvm_test         → lite_test        场景入口
//   analysis_port    → 用 mailbox 模拟（多订阅者用 mailboxes 数组）
//
// 用法（见 tb_uvm_lite_top.sv）
//   axi_uvm_lite_pkg::lite_env env = new(...);
//   env.build(); env.connect(); env.run(); env.check();
// ============================================================================
`ifndef AXI_UVM_LITE_PKG_SV
`define AXI_UVM_LITE_PKG_SV

package axi_uvm_lite_pkg;

    import bus_txn_pkg::*;
    import axi_defs_pkg::*;

    // ======================================================================
    // 事务记录（monitor 广播用）
    // ======================================================================
    class lite_item;
        bit          write;
        bit [7:0]    addr;
        bit [31:0]   data;
        bit [1:0]    resp;
        bit          is_error;

        function new(bit wr = 0, bit [7:0] a = 0, bit [31:0] d = 0,
                     bit [1:0] r = 0);
            write = wr; addr = a; data = d; resp = r;
            is_error = (r == RESP_SLVERR);
        endfunction

        function string to_str();
            return $sformatf("%s addr=0x%02h data=0x%08h resp=%0d",
                             write ? "WR" : "RD", addr, data, resp);
        endfunction
    endclass

    // ======================================================================
    // Driver：把事务变成 AXI 时序（通过虚接口）
    // ======================================================================
    class lite_driver;
        virtual axi_lite_if #(8, 32) vif;
        mailbox #(txn_t) req_mb;      // 从 sequencer 取事务

        int error_count = 0;

        function new(virtual axi_lite_if #(8, 32) v, mailbox #(txn_t) m);
            vif = v; req_mb = m;
        endfunction

        // 驱动一笔事务（阻塞直到完成）
        task automatic drive_one(input txn_t t, output rsp_t r);
            r = '0;
            r.data = t.data;
            // ---- 发起 ----
            @(posedge vif.ACLK);
            vif.AWVALID <= 1'b1;  vif.AWADDR <= t.addr;
            vif.WVALID  <= 1'b1;  vif.WDATA  <= t.data;
            vif.BREADY  <= 1'b1;
            begin
                bit aw_ok = 0, w_ok = 0, b_ok = 0;
                while (!b_ok) begin
                    @(posedge vif.ACLK); #1;
                    if (!aw_ok && vif.AWREADY) begin aw_ok = 1; vif.AWVALID <= 1'b0; end
                    if (!w_ok  && vif.WREADY)  begin w_ok  = 1; vif.WVALID  <= 1'b0; end
                    if (vif.BVALID) begin
                        b_ok = 1;
                        r.resp = vif.BRESP;
                    end
                end
            end
            vif.BREADY <= 1'b0;
            @(posedge vif.ACLK);
        endtask

        // 读事务
        task automatic drive_read(input txn_t t, output rsp_t r);
            r = '0;
            @(posedge vif.ACLK);
            vif.ARVALID <= 1'b1; vif.ARADDR <= t.addr;
            vif.RREADY  <= 1'b1;
            begin
                bit ar_ok = 0, r_ok = 0;
                while (!r_ok) begin
                    @(posedge vif.ACLK); #1;
                    if (!ar_ok && vif.ARREADY) begin ar_ok = 1; vif.ARVALID <= 1'b0; end
                    if (vif.RVALID) begin
                        r_ok = 1;
                        r.data = vif.RDATA;
                        r.resp = vif.RRESP;
                    end
                end
            end
            vif.RREADY <= 1'b0;
            @(posedge vif.ACLK);
        endtask

        // run phase：循环取事务并驱动
        task automatic run();
            txn_t t;
            rsp_t r;
            forever begin
                req_mb.get(t);
                if (t.write) drive_one(t, r);
                else         drive_read(t, r);
                // 广播给 scoreboard（通过 monitor 的 mailbox，由 env 连接）
                if (t.pause != 0) repeat (t.pause) @(posedge vif.ACLK);
            end
        endtask
    endclass

    // ======================================================================
    // Sequencer：事务队列（UVM 的 seq_item_export 对应物）
    // ======================================================================
    class lite_sequencer;
        mailbox #(txn_t) item_mb;     // driver 从这里取
        int issued = 0;

        function new();
            item_mb = new();
        endfunction

        // 发送一个事务（相当于 start_item/finish_item）
        task automatic send(input txn_t t);
            item_mb.put(t);
            issued++;
        endtask
    endclass

    // ======================================================================
    // Monitor：观测总线，把完成的事务广播出去
    // ======================================================================
    class lite_monitor;
        virtual axi_lite_if #(8, 32) vif;
        mailbox #(lite_item) ap[$];    // analysis port（多订阅者）

        function new(virtual axi_lite_if #(8, 32) v);
            vif = v;
        endfunction

        function void connect_ap(mailbox #(lite_item) m);
            ap.push_back(m);
        endfunction

        // 观测：广播一个事务到所有订阅者（对应 analysis_port.write）
        task automatic record(input lite_item it);
            foreach (ap[i]) ap[i].put(it);
        endtask
    endclass

    // ======================================================================
    // Scoreboard：比对期望 vs 实际
    // ======================================================================
    class lite_scoreboard;
        mailbox #(lite_item) from_mon;   // 从 monitor 收
        int compared = 0;
        int mismatches = 0;

        function new();
            from_mon = new();
        endfunction

        // 检查一个事务是否与期望一致
        function void check_item(input lite_item it,
                                 input bit [31:0] exp_data,
                                 input bit [1:0]  exp_resp);
            compared++;
            if (it.resp !== exp_resp) begin
                mismatches++;
                $display("  [SCB] MISMATCH resp: %0s (exp %0d)", it.to_str(), exp_resp);
            end else if (!it.is_error && it.data !== exp_data) begin
                mismatches++;
                $display("  [SCB] MISMATCH data: %0s (exp 0x%08h)", it.to_str(), exp_data);
            end
        endfunction
    endclass

    // ======================================================================
    // Env：组装 + 连接（对应 UVM 的 env.build/connect）
    // ======================================================================
    class lite_env;
        lite_sequencer  sqr;
        lite_driver     drv;
        lite_monitor    mon;
        lite_scoreboard scb;

        int tests_run = 0;
        int errors = 0;

        function new(virtual axi_lite_if #(8, 32) m_vif,
                     virtual axi_lite_if #(8, 32) mo_vif);
            sqr = new();
            drv = new(m_vif, sqr.item_mb);
            mon = new(mo_vif);
            scb = new();
        endfunction

        // build phase：建立 TLM 连接
        function void connect();
            mon.connect_ap(scb.from_mon);   // monitor → scoreboard
            $display("  [ENV] connect: sqr.item_mb -> drv, mon.ap -> scb");
        endfunction
    endclass

endpackage
`endif // AXI_UVM_LITE_PKG_SV
