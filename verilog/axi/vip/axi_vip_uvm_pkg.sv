// ============================================================================
// axi_vip_uvm_pkg：AXI4-Lite VIP 的 UVM 配套设施
// ============================================================================
// 用途：在装有 UVM 库的环境（irun/VCS/xsim）中，用标准 UVM 结构复用本 VIP。
//
// ⚠️ 本文件需要 UVM 库，本地（Verilator/slang 无 UVM）不编译。
//     本地验证请用同目录的非 UVM 版本：axi_vip_pkg.sv + axi_lite_vip.sv
//
// 编译方式（服务器）：
//   xrun -sv -uvm -uvmhome $UVM_HOME axi_vip_uvm_pkg.sv tb_uvm_top.sv
//   vcs -sverilog -ntb_opts uvm-1.2 axi_vip_uvm_pkg.sv tb_uvm_top.sv
//
// UVM 组件与本地 VIP 的对应：
//   本地                            UVM
//   ─────────────────────────────────────────────────────────
//   txn_t / rsp_t                   uvm_sequence_item
//   axi_sequence                    uvm_sequence
//   driver（drive/collect 任务）    uvm_driver
//   monitor（旁路采集）              uvm_monitor
//   axi_ref_model + check()         uvm_scoreboard
//   axi_lite_vip（env 组装）        uvm_env
//   report()                        uvm_report_server / analysis
// ============================================================================
`ifndef AXI_VIP_UVM_PKG_SV
`define AXI_VIP_UVM_PKG_SV

package axi_vip_uvm_pkg;

    import uvm_pkg::*;
    `include "uvm_macros.svh"
    import bus_txn_pkg::*;
    import axi_defs_pkg::*;

    // ==================================================================
    // 1. 事务对象（uvm_sequence_item）
    // ==================================================================
    class axi_txn extends uvm_sequence_item;
        rand bit        write;
        rand bit [7:0]  addr;
        rand bit [31:0] data;
        rand bit [7:0]  pause;
        rand bit        same;
        bit             exp_err;

        // 结果回填
        bit [31:0]      rdata;
        bit [1:0]       rresp;

        constraint c_addr_range { addr inside {[0:63]}; }

        `uvm_object_utils_begin(axi_txn)
            `uvm_field_int(write, UVM_ALL_ON)
            `uvm_field_int(addr,  UVM_ALL_ON)
            `uvm_field_int(data,  UVM_ALL_ON)
            `uvm_field_int(pause, UVM_ALL_ON)
            `uvm_field_int(same,  UVM_ALL_ON)
        `uvm_object_utils_end

        function new(string name = "axi_txn");
            super.new(name);
        endfunction

        function string convert2string();
            return $sformatf("%s addr=%02h data=%08h pause=%0d",
                             write ? "WR" : "RD", addr, data, pause);
        endfunction
    endclass

    // ==================================================================
    // 2. 序列
    // ==================================================================
    class axi_base_seq extends uvm_sequence #(axi_txn);
        `uvm_object_utils(axi_base_seq)

        function new(string name = "axi_base_seq");
            super.new(name);
        endfunction

        task body();
            repeat (20) begin
                req = axi_txn::type_id::create("req");
                start_item(req);
                if (!req.randomize())
                    `uvm_error("SEQ", "randomize failed")
                finish_item(req);
            end
        endtask
    endclass

    // ==================================================================
    // 3. Driver
    // ==================================================================
    class axi_driver extends uvm_driver #(axi_txn);
        `uvm_component_utils(axi_driver)
        virtual axi_lite_if vif;

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        task run_phase(uvm_phase phase);
            forever begin
                seq_item_port.get_next_item(req);
                drive_one(req);
                seq_item_port.item_done();
            end
        endtask

        task drive_one(axi_txn t);
            if (t.write) begin
                @(posedge vif.ACLK);
                vif.AWVALID <= 1'b1; vif.AWADDR <= t.addr;
                vif.WVALID  <= 1'b1; vif.WDATA  <= t.data;
                while (!(vif.AWREADY && vif.WREADY)) @(posedge vif.ACLK);
                vif.AWVALID <= 1'b0; vif.WVALID <= 1'b0;
            end else begin
                @(posedge vif.ACLK);
                vif.ARVALID <= 1'b1; vif.ARADDR <= t.addr;
                while (!vif.ARREADY) @(posedge vif.ACLK);
                vif.ARVALID <= 1'b0;
            end
            if (t.pause != 0) repeat (t.pause) @(posedge vif.ACLK);
        endtask
    endclass

    // ==================================================================
    // 4. Monitor（旁路观测，不干扰 DUT）
    // ==================================================================
    class axi_monitor extends uvm_monitor;
        `uvm_component_utils(axi_monitor)
        virtual axi_lite_if vif;
        uvm_analysis_port #(axi_txn) ap;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            ap = new("ap", this);
        endfunction

        task run_phase(uvm_phase phase);
            forever begin
                axi_txn t;
                @(posedge vif.ACLK);
                // 采集写事务
                if (vif.AWVALID && vif.AWREADY) begin
                    t = axi_txn::type_id::create("mon_wr");
                    t.write = 1'b1; t.addr = vif.AWADDR;
                    ap.write(t);
                end
                // 采集读事务
                if (vif.ARVALID && vif.ARREADY) begin
                    t = axi_txn::type_id::create("mon_rd");
                    t.write = 1'b0; t.addr = vif.ARADDR;
                    ap.write(t);
                end
            end
        endtask
    endclass

    // ==================================================================
    // 5. Scoreboard
    // ==================================================================
    class axi_scoreboard extends uvm_scoreboard;
        `uvm_component_utils(axi_scoreboard)
        uvm_analysis_imp #(axi_txn, axi_scoreboard) imp;
        bit [31:0] mem[bit [31:0]];
        int checks = 0, errors = 0;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            imp = new("imp", this);
        endfunction

        function void write(axi_txn t);
            checks++;
            if (t.write)
                mem[t.addr >> 2] = t.data;      // 字寻址，与 DUT 对齐
            else if (t.rdata !== ((mem.exists(t.addr >> 2)) ? mem[t.addr >> 2] : '0)) begin
                `uvm_error("SB", $sformatf("RD addr=%02h mismatch", t.addr))
                errors++;
            end
        endfunction

        function void report_phase(uvm_phase phase);
            `uvm_info("SB", $sformatf("checks=%0d errors=%0d", checks, errors), UVM_LOW)
        endfunction
    endclass

    // ==================================================================
    // 6. Env（组装）
    // ==================================================================
    class axi_vip_env extends uvm_env;
        `uvm_component_utils(axi_vip_env)
        axi_driver     drv;
        axi_monitor    mon;
        axi_scoreboard sb;

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            drv = axi_driver::type_id::create("drv", this);
            mon = axi_monitor::type_id::create("mon", this);
            sb  = axi_scoreboard::type_id::create("sb", this);
        endfunction

        function void connect_phase(uvm_phase phase);
            mon.ap.connect(sb.imp);
        endfunction
    endclass

endpackage

`endif // AXI_VIP_UVM_PKG_SV