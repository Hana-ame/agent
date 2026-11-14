// ============================================================================
// tb_uvm_top：UVM 环境顶层接入示例（需 UVM 库，本地不编译）
// ============================================================================
// 服务器运行：
//   xrun -sv -uvm -uvmhome $UVM_HOME axi_vip_uvm_pkg.sv tb_uvm_top.sv
`ifndef TB_UVM_TOP_SV
`define TB_UVM_TOP_SV

`include "uvm_macros.svh"
import uvm_pkg::*;
import axi_vip_uvm_pkg::*;

module tb_uvm_top;
    logic ACLK = 0, ARESETn = 0;
    always #5 ACLK = ~ACLK;

    axi_lite_if #(.AW(8), .DW(32)) axi(ACLK);
    apb_if      #(.AW(8), .DW(32)) apb(ACLK);

    // DUT（与本地版本相同）
    axi_apb_bridge_sv #(.AW(8), .DW(32)) bridge (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
    );
    apb_slave_regs_sv #(.AW(8), .DW(32), .NREG(16), .WAIT_CYCLES(2)) slave (
        .PCLK(ACLK), .PRESETn(ARESETn), .apb(apb)
    );

    initial begin
        uvm_config_db #(virtual axi_lite_if)::set(null, "*drv", "vif", axi);
        uvm_config_db #(virtual axi_lite_if)::set(null, "*mon", "vif", axi);
        run_test("axi_test");
    end

    task automatic run_test(string name);
        uvm_root::get().run_test(name);
    endtask
endmodule

// UVM test
class axi_test extends uvm_test;
    `uvm_component_utils(axi_test)
    axi_vip_env env;

    function new(string name, uvm_component parent);
        super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        env = axi_vip_env::type_id::create("env", this);
    endfunction

    task run_phase(uvm_phase phase);
        axi_base_seq seq;
        phase.raise_objection(this);
        seq = axi_base_seq::type_id::create("seq");
        seq.start(env.sqr);
        phase.drop_objection(this);
    endtask
endclass

`endif // TB_UVM_TOP_SV