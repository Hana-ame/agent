// 完整 SystemVerilog testbench：AXI4-Lite → APB → 慢速外设 → 响应回 AXI
// 采用「数据向量 → 激励向量 → 数据向量」完整测试闭环：
//
//   ┌─────────────────────────────────────────────────────────────────┐
//   │ ① 数据向量集（stim_vecs，输入知识库）                            │
//   │    op(WR/RD) + addr + data + exp_resp                           │
//   │    ├─ 程序化向量：写 NREG 条 → 读 NREG 条（回环基本集）          │
//   │    ├─ 随机向量：200 条（写/读/同发写混合）                       │
//   │    └─ 边界向量：越界读写（期望 SLVERR）                          │
//   └──────────────┬──────────────────────────────────────────────────┘
//                  ▼
//   ┌─────────────────────────────────────────────────────────────────┐
//   │ ② 激励向量驱动（drive_vec）                                      │
//   │    逐向量把 AXI 事务打到 DUT（interface + 时钟同步）              │
//   │    返回捕获的 data + resp（DUT 的实际输出）                       │
//   └──────────────┬──────────────────────────────────────────────────┘
//                  ▼
//   ┌─────────────────────────────────────────────────────────────────┐
//   │ ③ 数据向量比对（scoreboard）                                     │
//   │    期望 = ref_vec（参考模型镜像） vs 实际 = captured              │
//   │    - 写向量：更新 ref_mem（期望数据向量集）                       │
//   │    - 读向量：captured.data !== ref_mem[addr] → FAIL               │
//   │    - 越界：captured.resp !== exp_resp(SLVERR) → FAIL              │
//   └─────────────────────────────────────────────────────────────────┘
//    反向闭环：❸ 更新的期望 ref_mem 正是 ❶ 后续读向量的比对基准
//
// 验证工具链：slang（编译合法）+ Verilator ≥5.055（实际仿真）
`ifndef TB_AXI_APB_SV_SV
`define TB_AXI_APB_SV_SV
`timescale 1ns / 1ps

module tb_axi_apb_sv;

    localparam int AW  = 8;
    localparam int DW  = 32;
    localparam int NREG = 16;
    localparam int WAIT_CYCLES = 2;
    localparam int NTXN = 200;   // 随机阶段事务数

    // ---- 时钟/复位 ----
    logic ACLK = 0;
    logic ARESETn = 0;
    always #5 ACLK = ~ACLK;

    // ---- 接口 ----
    axi_lite_if #(.AW(AW), .DW(DW)) axi(ACLK);
    apb_if      #(.AW(AW), .DW(DW)) apb(ACLK);

    // ---- DUT：转换设计 + 慢速从机（接口端口连接） ----
    axi_apb_bridge_sv #(.AW(AW), .DW(DW)) bridge (
        .ACLK   (ACLK),
        .ARESETn(ARESETn),
        .axi    (axi),   // slave modport
        .apb    (apb)    // master modport
    );

    apb_slave_regs_sv #(.AW(AW), .DW(DW), .NREG(NREG),
                        .WAIT_CYCLES(WAIT_CYCLES)) slave (
        .PCLK   (ACLK),
        .PRESETn(ARESETn),
        .apb    (apb)     // slave modport
    );

    // ==================================================================
    // ① 数据向量集（单一事实来源：输入知识库 + 期望基准）
    // ==================================================================
    // 数据向量：一条 = 一个 AXI 事务的完整描述（含期望）
    typedef struct packed {
        bit            write;     // 1=写 0=读
        bit            same;      // 写时 AW+W 同拍
        bit [AW-1:0]   addr;
        bit [DW-1:0]   data;      // 写数据（读时忽略）
        bit [1:0]      exp_resp;  // 期望响应（OKAY / SLVERR）
        bit            valid;     // 向量有效
    } data_vec_t;

    // 激励向量集：先写后读回环 + 随机 + 越界（顺序即执行顺序）
    data_vec_t stim_vecs[$];

    // 期望数据向量集：参考模型镜像
    //   ref_mem[addr] = 该地址应读回的数据（由写向量更新，读向量比对）
    logic [DW-1:0] ref_mem [0:NREG-1];

    // ---- 数据向量构造辅助 ----
    function automatic data_vec_t make_vec(input bit wr, input bit same,
                                           input [AW-1:0] a, input [DW-1:0] d,
                                           input [1:0] resp);
        data_vec_t v;
        v = '0;
        v.write   = wr;
        v.same    = same;
        v.addr    = a;
        v.data    = d;
        v.exp_resp = resp;
        v.valid   = 1'b1;
        return v;
    endfunction

    // ---- 构建激励向量集 ----
    int vec_cnt = 0;
    initial begin
        data_vec_t v;
        // ①a 回环基本集：写 0..NREG-1（random 数据）→ 读 0..NREG-1
        for (int i = 0; i < NREG; i++) begin
            v = make_vec(1'b1, 1'b0, i[AW-1:0]*4, $urandom, axi_defs_pkg::RESP_OKAY);
            stim_vecs.push_back(v);
        end
        for (int i = 0; i < NREG; i++) begin
            v = make_vec(1'b0, 1'b0, i[AW-1:0]*4, '0, axi_defs_pkg::RESP_OKAY);
            stim_vecs.push_back(v);
        end
        // ①b 随机向量：NTXN 条（写/读/同发写混合，addr 界内）
        for (int k = 0; k < NTXN; k++) begin
            automatic bit wr = $urandom & 1;
            automatic bit same = wr ? ($urandom & 1) : 0;
            automatic bit [AW-1:0] a = $urandom % (NREG*4);
            v = make_vec(wr, same, a, $urandom, axi_defs_pkg::RESP_OKAY);
            stim_vecs.push_back(v);
        end
        // ①c 边界向量：越界读写（期望 SLVERR）
        v = make_vec(1'b1, 1'b0, NREG*4, 32'hDEAD_BEEF, axi_defs_pkg::RESP_SLVERR);
        stim_vecs.push_back(v);
        v = make_vec(1'b0, 1'b0, NREG*4, '0, axi_defs_pkg::RESP_SLVERR);
        stim_vecs.push_back(v);
        vec_cnt = stim_vecs.size();
        $display("== ① data vectors: %0d built (loopback %0d + random %0d + oob 2) ==",
                 vec_cnt, NREG*2, NTXN);
    end

    // ==================================================================
    // ② 激励向量驱动：逐向量打 AXI 事务，捕获 DUT 实际输出
    // ==================================================================
    class AxiDriver;
        virtual axi_lite_if #(.AW(AW), .DW(DW)) vif;

        function new();
            // 无参构造，vif 由外部直接赋值（兼容 Verilator 静态绑定）
        endfunction

        // 执行一条数据向量 → 返回捕获的数据向量（data/resp 填实际值）
        task run_vec(input data_vec_t v, output data_vec_t captured);
            captured = v;   // 继承描述字段
            if (v.write) begin
                if (v.same) begin
                    // 同发写：AW+W 一拍
                    @(posedge vif.ACLK);
                    vif.AWVALID <= 1'b1; vif.AWADDR <= v.addr;
                    vif.WVALID  <= 1'b1; vif.WDATA  <= v.data;
                    while (!(vif.AWREADY && vif.WREADY)) @(posedge vif.ACLK);
                    vif.AWVALID <= 1'b0; vif.WVALID <= 1'b0;
                end else begin
                    // 分开发：AW 先，W 后（握手判据 VALID&&READY）
                    @(posedge vif.ACLK);
                    vif.AWVALID <= 1'b1; vif.AWADDR <= v.addr;
                    do @(posedge vif.ACLK); while (!(vif.AWVALID && vif.AWREADY));
                    vif.AWVALID <= 1'b0;
                    @(posedge vif.ACLK);
                    vif.WVALID <= 1'b1; vif.WDATA <= v.data;
                    do @(posedge vif.ACLK); while (!(vif.WVALID && vif.WREADY));
                    vif.WVALID <= 1'b0;
                end
                // B 响应
                vif.BREADY <= 1'b1;
                do @(posedge vif.ACLK); while (!vif.BVALID);
                captured.data = v.data;
                captured.exp_resp = vif.BRESP;   // 捕获实际响应
                vif.BREADY <= 1'b0;
                @(posedge vif.ACLK);
            end else begin
                // 读：AR → R
                @(posedge vif.ACLK);
                vif.ARVALID <= 1'b1; vif.ARADDR <= v.addr;
                while (!vif.ARREADY) @(posedge vif.ACLK);
                vif.ARVALID <= 1'b0;
                vif.RREADY <= 1'b1;
                do @(posedge vif.ACLK); while (!vif.RVALID);
                captured.data    = vif.RDATA;    // 捕获实际读回数据
                captured.exp_resp = vif.RRESP;   // 捕获实际响应
                vif.RREADY <= 1'b0;
                @(posedge vif.ACLK);
            end
        endtask
    endclass

    // ==================================================================
    // ③ 数据向量比对（scoreboard）：期望 vs 实际
    // ==================================================================
    int errors = 0;
    int checks = 0;

    // 写向量：更新期望数据向量集（写成功后 ref_mem[addr]=data）
    function automatic void sb_write_vector(input data_vec_t exp_v,
                                            input data_vec_t cap_v);
        int idx = exp_v.addr >> 2;
        checks++;
        if (cap_v.exp_resp !== axi_defs_pkg::RESP_OKAY) begin
            // 期望 OKAY 却得错误 → FAIL；期望 SLVERR（越界）→ 正确
            if (idx < NREG) begin
                $display("SB FAIL[%0d]: WR addr=%0d resp=%s (expect OKAY)",
                         checks, exp_v.addr, axi_defs_pkg::resp_name(cap_v.exp_resp));
                errors++;
            end
        end else begin
            if (idx >= NREG) begin
                $display("SB FAIL[%0d]: OOB WR addr=%0d got OKAY", checks, exp_v.addr);
                errors++;
            end else
                ref_mem[idx] = exp_v.data;   // 更新期望向量集（反向闭环）
        end
    endfunction

    // 读向量：比对 ref_mem（期望） vs 捕获数据（实际）
    function automatic void sb_read_vector(input data_vec_t exp_v,
                                           input data_vec_t cap_v);
        int idx = exp_v.addr >> 2;
        checks++;
        if (cap_v.exp_resp !== axi_defs_pkg::RESP_OKAY) begin
            if (idx < NREG) begin
                $display("SB FAIL[%0d]: RD addr=%0d resp=%s (expect OKAY)",
                         checks, exp_v.addr, axi_defs_pkg::resp_name(cap_v.exp_resp));
                errors++;
            end
        end else begin
            if (idx >= NREG) begin
                $display("SB FAIL[%0d]: OOB RD addr=%0d got OKAY", checks, exp_v.addr);
                errors++;
            end else if (cap_v.data !== ref_mem[idx]) begin
                $display("SB FAIL[%0d]: RD addr=%0d exp=%h got=%h", checks,
                         exp_v.addr, ref_mem[idx], cap_v.data);
                errors++;
            end
        end
    endfunction

    // ==================================================================
    // 主流程：数据向量 → 激励向量 → 数据向量 闭环执行
    // ==================================================================
    AxiDriver drv;
    int txns = 0;

    initial begin
        drv = new();
        drv.vif = axi;   // 虚接口直接赋值

        // 复位
        ARESETn = 0;
        repeat (4) @(posedge ACLK);
        ARESETn = 1;

        $display("== SV testbench: AXI->APB->AXI vector loop (WAIT=%0d) ==", WAIT_CYCLES);

        // 等待向量集构建完成（同 initial，构建块的 @ 先于本块执行）
        // （构建块在最前声明，先于本 initial 完成；此处无额外同步需要）
        $display("== ② 激励向量驱动: %0d vectors over AXI ==", vec_cnt);

        // 逐向量：激励驱动 → 捕获 → 比对
        foreach (stim_vecs[i]) begin
            data_vec_t cap;
            drv.run_vec(stim_vecs[i], cap);
            if (stim_vecs[i].write)
                sb_write_vector(stim_vecs[i], cap);
            else
                sb_read_vector(stim_vecs[i], cap);
            txns++;
        end

        // ---- 汇总 ----
        $display("== ③ 数据向量比对: %0d checks, %0d errors ==", checks, errors);
        $display("-- SUMMARY: %0d txns driven, ref_mem snapshot:", txns);
        for (int i = 0; i < NREG; i += 4) begin
            $display("    [%02h] %h %h %h %h", i*4,
                     ref_mem[i], ref_mem[i+1], ref_mem[i+2], ref_mem[i+3]);
        end
        if (errors == 0) begin
            $display("PASS: axi->apb->axi SV vector testbench (%0d vectors)", txns);
            $finish;
        end else begin
            $display("FAIL: %0d errors in %0d checks", errors, checks);
            $fatal(1, "SV vector testbench failed");
        end
    end

    // ---- 波形 ----
    initial begin
        $dumpfile("tb_axi_apb_sv.vcd");
        $dumpvars(0, tb_axi_apb_sv);
    end

endmodule
`endif // TB_AXI_APB_SV_SV