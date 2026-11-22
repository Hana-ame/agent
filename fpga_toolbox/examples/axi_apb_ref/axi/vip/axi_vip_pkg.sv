// ============================================================================
// axi_vip_pkg：AXI4-Lite 验证 IP（VIP）——打激励 + 验证激励
// ============================================================================
// 设计目标：一个模块，可直接接入任意 AXI4-Lite DUT 的验证。
//   只依赖 axi_lite_if 接口，不依赖具体 DUT 内部结构。
//
// 组成（经典 UVM 结构的轻量化，不依赖 UVM 库，本地可用 Verilator/slang 验证）：
//
//   ┌──────────────────────────────────────────────────────────────┐
//   │ ① sequence  —— 激励产生：连笔事务（含 pause/same/exp_err）    │
//   │ ② driver    —— 打激励：把 txn_t 展开成 AXI 握手时序          │
//   │ ③ monitor   —— 采集：旁路观测总线，重建出事务（不干扰 DUT）   │
//   │ ④ scoreboard—— 验证激励：参考模型逐事务比对，输出 PASS/FAIL  │
//   │ ⑤ env       —— 组装：把 ①~④ 装进一个可直接实例化的模块       │
//   └──────────────────────────────────────────────────────────────┘
//
// 接入方式（三行）：
//     axi_lite_vip #(.AW(8), .DW(32)) vip (
//         .ACLK(clk), .ARESETn(rst_n), .axi(axi_if)
//     );
//     ... 运行 ...
//     vip.report();     // 打印 PASS/FAIL 与统计
//
// 与 UVM 的对应关系（见 README）：
//     sequence→uvm_sequence  driver→uvm_driver
//     monitor →uvm_monitor   scoreboard→uvm_scoreboard
//     env     →uvm_env       report→uvm_report_server
// ============================================================================
`ifndef AXI_VIP_PKG_SV
`define AXI_VIP_PKG_SV
`timescale 1ns / 1ps

package axi_vip_pkg;

    import bus_txn_pkg::*;
    import axi_defs_pkg::*;

    // ==================================================================
    // 事务（复用第 3 层结构体，VIP 不再定义自己的事务）
    //   txn_t  —— 激励
    //   rsp_t  —— 结果
    // ==================================================================

    // 参考模型：AXI 侧事务 → 期望响应/数据
    class axi_ref_model;
        // 存储镜像：地址 → 数据（联动 APB 从机的寄存器堆语义）
        bit [31:0]        mem[bit [31:0]];
        bit               has_mem = 0;   // 是否启用镜像（由 scoreboard 配置）
        bit               word_addressed = 1;  // 1=按字(addr>>2)寻址，0=按字节寻址

        function new();
        endfunction

        // 把事务地址转成存储键（与 DUT 的地址译码语义对齐）
        function bit [31:0] key(input bit [31:0] addr);
            return word_addressed ? (addr >> 2) : addr;
        endfunction

        // 预测一个事务的期望结果
        //   addr 越界（由 limit 给出）→ 期望 SLVERR
        function rsp_t predict(input txn_t t, input bit [31:0] limit);
            rsp_t r;
            bit [31:0] k;
            k = key(t.addr);
            r = '0;
            if (t.addr >= limit) begin
                r.resp = RESP_SLVERR;      // 越界
                r.data = '0;
            end else if (t.write) begin
                r.resp = RESP_OKAY;
                r.data = t.data;
                if (has_mem) mem[k] = t.data;
            end else begin
                r.resp = RESP_OKAY;
                r.data = (has_mem && mem.exists(k)) ? mem[k] : '0;
            end
            return r;
        endfunction

        // 仅做响应预测（不含数据，monitor 用）
        function bit [1:0] predict_resp(input bit [31:0] addr, input bit [31:0] limit);
            return (addr >= limit) ? RESP_SLVERR : RESP_OKAY;
        endfunction
    endclass

    // ==================================================================
    // 激励序列：组织一串事务
    // ==================================================================
    class axi_sequence;
        txn_t q[$];

        function new();
        endfunction

        // 添加一笔事务
        function void add(input bit wr, input bit [7:0] a, input bit [31:0] d,
                          input bit [7:0] pause = 0, input bit same = 0,
                          input bit exp_err = 0);
            txn_t t = '{write: wr, addr: a, data: d, pause: pause,
                        same: same, exp_err: exp_err};
            q.push_back(t);
        endfunction

        // 写后读回（回环）
        function void add_loopback(input bit [7:0] a, input bit [31:0] d,
                                   input bit [7:0] pause = 1);
            add(1'b1, a, d, pause);
            add(1'b0, a, '0, 0);
        endfunction

        // 随机序列
        function void add_random(input int n, input bit [31:0] limit);
            for (int i = 0; i < n; i++) begin
                automatic bit wr = $urandom & 1;
                automatic bit same = wr ? ($urandom & 1) : 0;
                automatic bit [7:0] pz = $urandom % 4;
                automatic bit [7:0] a = $urandom % limit;
                add(wr, a, $urandom, pz, same);
            end
        endfunction

        function int size();
            return q.size();
        endfunction
    endclass

    // ==================================================================
    // 统计与检查
    // ==================================================================
    class axi_vip_stats;
        int txns       = 0;
        int checks     = 0;
        int errors     = 0;
        int pause_total = 0;
        int reads      = 0;
        int writes     = 0;
        int slverr_cnt = 0;

        function void dump(string tag = "VIP");
            $display("[%0s] txns=%0d (W=%0d R=%0d) checks=%0d errors=%0d pause=%0d SLVERR=%0d",
                     tag, txns, writes, reads, checks, errors, pause_total, slverr_cnt);
        endfunction
    endclass

endpackage

`endif // AXI_VIP_PKG_SV