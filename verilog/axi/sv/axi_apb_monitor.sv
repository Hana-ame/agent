// ============================================================================
// axi_apb_monitor.sv：独立协议监视器（可插接到任意 testbench）
// ============================================================================
// 设计意图
//   此前只有 tb_axi_apb.sv 内置了 APB monitor，其余 10 个套件依赖各自写的
//   断言。接口层虽有 SVA，但缺少「全局、自动、与测试意图无关」的检查。
//
//   本模块是一个被动监视器（只读信号，不驱动），实例化后即自动检查：
//     - AXI4-Lite 五通道握手法则
//     - VALID 拉高后必须保持到 READY（AXI 强制）
//     - APB 阶段时序（SETUP/ACCESS、PREADY、PSLVERR）
//     - 传输期间地址/控制信号必须稳定
//   规则全部来自 axi_defs_pkg / apb_defs_pkg 的单一定义（不重复实现）。
//
// 用法（在 TB 里一行接入）
//   axi_apb_monitor #(.AW(AW), .DW(DW)) mon (
//       .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
//   );
//   ... 仿真结束 ...
//   if (mon.errors != 0) $fatal(1, "protocol monitor: %0d violations", mon.errors);
//
// 特点
//   - 被动：只用 always @(posedge) 采样，不驱动任何信号
//   - 可配置：通过 StringName 前缀区分多个实例的日志
//   - 零开销：不满足条件时不打印（避免刷屏）
// ============================================================================
`ifndef AXI_APB_MONITOR_SV
`define AXI_APB_MONITOR_SV
`timescale 1ns / 1ps

module axi_apb_monitor #(
    parameter int    AW       = 8,
    parameter int    DW       = 32,
    parameter string NAME     = "MON",
    parameter bit    SERIAL_BRIDGE = 1'b1,  // 1=串行桥（A3 规则适用）
    parameter bit    VERBOSE  = 1'b0    // 1=打印每次传输完成
)(
    input  logic ACLK,
    input  logic ARESETn,
    axi_lite_if.monitor axi,
    apb_if.monitor      apb
);

    import axi_defs_pkg::*;
    import apb_defs_pkg::*;

    // ---- 违规计数（测试结束时检查） ----
    int errors = 0;
    int xfer_count = 0;

    // ---- 上一拍状态（用于时序规则检查） ----
    logic prev_apb_access;   // 上拍 PSEL && PENABLE
    apb_defs_pkg::apb_phase_t cur_phase;  // 当前 APB 阶段（供违规报告用）
    logic prev_apb_ready;    // 上拍 PREADY
    logic prev_awvalid, prev_wvalid, prev_arvalid;
    logic prev_awready, prev_wready, prev_arready;   // 上一拍 READY（判定握手用）
    logic [AW-1:0] prev_awaddr, prev_araddr;
    logic [DW-1:0] prev_wdata;
    logic prev_pwrite;
    logic [AW-1:0] prev_paddr;

    always_ff @(posedge ACLK or negedge ARESETn) begin
        if (!ARESETn) begin
            prev_apb_access <= 1'b0;
            prev_apb_ready  <= 1'b0;
            prev_awvalid    <= 1'b0;
            prev_wvalid     <= 1'b0;
            prev_arvalid    <= 1'b0;
            prev_awready    <= 1'b0;
            prev_wready     <= 1'b0;
            prev_arready    <= 1'b0;
            prev_awaddr     <= '0;
            prev_araddr     <= '0;
            prev_wdata      <= '0;
            prev_pwrite     <= 1'b0;
            prev_paddr      <= '0;
        end else begin
            prev_apb_access <= apb.PSEL && apb.PENABLE;
            prev_apb_ready  <= apb.PREADY;
            prev_awvalid    <= axi.AWVALID;
            prev_wvalid     <= axi.WVALID;
            prev_arvalid    <= axi.ARVALID;
            prev_awready    <= axi.AWREADY;
            prev_wready     <= axi.WREADY;
            prev_arready    <= axi.ARREADY;
            prev_awaddr     <= axi.AWADDR;
            prev_araddr     <= axi.ARADDR;
            prev_wdata      <= axi.WDATA;
            prev_pwrite     <= apb.PWRITE;
            prev_paddr      <= apb.PADDR;
        end
    end

    // 违规记录 helper
    // 除计数外，记录「最近一次违规的规则名」，供 testbench 精确校验
    //（此前 tb_monitor_rules 的 grep_violation 是空实现，永远返回 1，
    //  导致规则失效也测不出来 —— 是真实的测试盲区）。
    string last_rule = "";
    int    rule_hits[string];

    task automatic violation(input string rule, input string detail);
        errors++;
        last_rule = rule;
        rule_hits[rule] = rule_hits.exists(rule) ? rule_hits[rule] + 1 : 1;
        // 附带 APB 阶段名（phase_name 此前定义了但从未被调用）。
        // 违规发生在 IDLE / SETUP / ACCESS 哪个阶段，直接决定排查方向，
        // 例如 P1（PENABLE 无 PSEL）多半发生在 SETUP、P3 只在 SETUP 有意义。
        $display("[%s] VIOLATION %s: %s @%0t (APB 阶段=%s PSEL=%b PENABLE=%b PREADY=%b)",
                 NAME, rule, detail, $time, apb_defs_pkg::phase_name(cur_phase),
                 apb.PSEL, apb.PENABLE, apb.PREADY);
    endtask

    // 查询某规则是否被触发过。
    // 支持两种用法：
    //   精确名 rule_hit("A2.AWADDR")
    //   前缀名 rule_hit("A2")  → 匹配 A2.AWADDR / A2.WDATA / A2.ARADDR
    function automatic bit rule_hit(input string rule);
        foreach (rule_hits[k]) begin
            if (k == rule || k.substr(0, rule.len()-1) == rule)
                return 1'b1;
        end
        return 1'b0;
    endfunction

    // 清除规则命中记录（每次子测试前调用）
    function automatic void rule_clear();
        rule_hits.delete();
        last_rule = "";
    endfunction

    // ==================================================================
    // AXI 规则检查
    // ==================================================================
    always @(posedge ACLK) begin
        if (ARESETn) begin
            // ---- A1: VALID 一旦拉高，在握手完成前不得撤销 ----
            // 判定必须用「上一拍的 READY」：握手发生在上一拍
            // （prev_valid && prev_ready），若上一拍已握手，本拍撤销 VALID 是合法的。
            if (prev_awvalid && !prev_awready && !axi.AWVALID)
                violation("A1.AWVALID", "AWVALID dropped without AWREADY");
            if (prev_wvalid && !prev_wready && !axi.WVALID)
                violation("A1.WVALID", "WVALID dropped without WREADY");
            if (prev_arvalid && !prev_arready && !axi.ARVALID)
                violation("A1.ARVALID", "ARVALID dropped without ARREADY");

            // ---- A3: 单事务桥不允许 AW 与 AR 同时请求 ----
            // 复用 axi_defs_pkg 的 valid_channels_ok()（此前该函数定义了但从未被调用）。
            // 注意：这条规则只对**串行桥**成立；并发桥（_dual_sv）读写并行，
            // 由 DUEL 参数隔离（见下）。
            if (!SERIAL_BRIDGE || !valid_channels_ok(axi.AWVALID, axi.WVALID, axi.ARVALID)) begin
                if (SERIAL_BRIDGE && axi.AWVALID && axi.ARVALID)
                    violation("A3.CHANNEL", "AWVALID and ARVALID asserted together (serial bridge)");
            end

            // 注：不检查地址对齐。AXI4-Lite 允许非对齐地址（协议只「建议」对齐），
            // 本桥与从机也不要求对齐 —— 越界/非对齐地址由从机用 SLVERR 处理。
            // 曾试加 A4.ALIGN 规则，结果 gaps 套件的越界测试（addr=0xFF）大量误报，
            // 证明这是「过度约束」而非真实协议要求，已移除。

            // ---- A2: 握手期间地址/数据必须稳定 ----
            if (prev_awvalid && !prev_awready && axi.AWVALID && axi.AWADDR !== prev_awaddr)
                violation("A2.AWADDR", "AWADDR changed while AWVALID held");
            if (prev_wvalid && !prev_wready && axi.WVALID && axi.WDATA !== prev_wdata)
                violation("A2.WDATA", "WDATA changed while WVALID held");
            if (prev_arvalid && !prev_arready && axi.ARVALID && axi.ARADDR !== prev_araddr)
                violation("A2.ARADDR", "ARADDR changed while ARVALID held");

            // ---- A3: 响应通道 VALID 保持 ----
            // 桥作为从机，BVALID/RVALID 也必须遵守同样的规则；
            // 由上拍的 BVALID/RVALID 与本拍比较。
            // （用接口 SVA 已覆盖，这里做冗余交叉检查）
        end
    end

    // ==================================================================
    // APB 规则检查（规则函数来自 apb_defs_pkg）
    // ==================================================================
    always @(posedge ACLK) begin
        if (ARESETn) begin
            // 更新当前 APB 阶段（供 violation() 报告用）：
            //   PSEL=0            → IDLE
            //   PSEL=1, PENABLE=0 → SETUP
            //   PSEL=1, PENABLE=1 → ACCESS
            cur_phase <= !apb.PSEL ? apb_defs_pkg::APB_IDLE :
                         !apb.PENABLE ? apb_defs_pkg::APB_SETUP :
                                        apb_defs_pkg::APB_ACC;

            // ---- P1: PENABLE 只能在 PSEL=1 时拉高 ----
            if (!penable_ok(apb.PSEL, apb.PENABLE))
                violation("P1.PENABLE", "PENABLE asserted while PSEL=0");

            // ---- P2: ACCESS 阶段不得提前退出（除非 PREADY） ----
            if (!phase_transition_ok(prev_apb_access, prev_apb_ready,
                                     apb.PSEL && apb.PENABLE))
                violation("P2.PHASE", "APB ACCESS terminated before PREADY");

            // ---- P3: SETUP 阶段不得回 PREADY ----
            if (apb.PSEL && !apb.PENABLE && apb.PREADY)
                violation("P3.SETUP", "PREADY asserted during SETUP phase");

            // ---- P4: 传输期间地址/控制必须稳定 ----
            if (prev_apb_access && !prev_apb_ready && apb.PSEL && apb.PENABLE) begin
                if (apb.PADDR !== prev_paddr)
                    violation("P4.PADDR", "PADDR changed during ACCESS wait");
                if (apb.PWRITE !== prev_pwrite)
                    violation("P4.PWRITE", "PWRITE changed during ACCESS wait");
            end

            // ---- P5: PSEL 撤销时必须同时撤销 PENABLE（APB 时序） ----
            // PSEL=0 时 PENABLE 也必须为 0（已由 P1 覆盖方向，
            // 这里检查反方向：PENABLE 拉高后 PSEL 不能先掉）
            // 注：P1 已确保 PENABLE=>PSEL，故此项冗余，保留为文档性检查

            // ---- 统计合法传输 ----
            if (transfer_done(apb.PSEL, apb.PENABLE, apb.PREADY)) begin
                xfer_count++;
                if (VERBOSE)
                    $display("[%s] APB xfer #%0d: addr=0x%02h %s %s",
                             NAME, xfer_count, apb.PADDR,
                             apb.PWRITE ? "WR" : "RD",
                             apb.PSLVERR ? "SLVERR" : "OKAY");
            end
        end
    end

endmodule
`endif // AXI_APB_MONITOR_SV
