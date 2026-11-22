// AXI4-Lite 接口（完整 SV 版）
// 供完整 SV 链路（interface 端口版）使用。iverilog 不可编译，需 slang 检查
// 或 irun/VCS 运行。
//
// 包含：
//   - AXI4-Lite 五通道信号（AW/W/B/AR/R）
//   - modport: master / slave（方向归组）
//   - clocking: cb（同步采样/驱动，避免竞争）
//   - SVA 断言: 握手法则（VALID 与 READY 同拍完成时本拍赋值；主从双方
//     VALID 不能在没有 VALID 的通道上拉高 READY —— 用握手完成断言表达）
`ifndef AXI_LITE_IF_SV
`define AXI_LITE_IF_SV

interface axi_lite_if #(
    parameter AW = 8,
    parameter DW = 32
) (input logic ACLK);

    // ---- 全局信号（AXI 规范） ----
    // ACLK 由外部提供；ARESETn 低有效复位

    // ---- 写地址通道 ----
    logic        AWVALID;
    logic        AWREADY;
    logic [AW-1:0] AWADDR;

    // ---- 写数据通道 ----
    logic        WVALID;
    logic        WREADY;
    logic [DW-1:0] WDATA;

    // ---- 写响应通道 ----
    logic        BVALID;
    logic        BREADY;
    logic [1:0]  BRESP;

    // ---- 读地址通道 ----
    logic        ARVALID;
    logic        ARREADY;
    logic [AW-1:0] ARADDR;

    // ---- 读数据通道 ----
    logic        RVALID;
    logic        RREADY;
    logic [DW-1:0] RDATA;
    logic [1:0]  RRESP;

    // ---- modport ----
    modport master (
        output AWVALID, AWADDR, WVALID, WDATA, BREADY,
               ARVALID, ARADDR, RREADY,
        input  AWREADY, WREADY, BVALID, BRESP, ARREADY, RVALID, RDATA, RRESP
    );
    modport slave (
        input  AWVALID, AWADDR, WVALID, WDATA, BREADY,
               ARVALID, ARADDR, RREADY,
        output AWREADY, WREADY, BVALID, BRESP, ARREADY, RVALID, RDATA, RRESP
    );
    // 监视器：只读全部信号（被动检查协议合法性）
    modport monitor (
        input AWVALID, AWADDR, WVALID, WDATA, BREADY,
              AWREADY, WREADY, BVALID, BRESP,
              ARVALID, ARADDR, RREADY, ARREADY, RVALID, RDATA, RRESP
    );

    // ---- clocking block（同步驱动/采样） ----
    // ⚠ 状态说明：本 clocking block **当前没有任何 TB 使用**。
    //   所有 TB 都用直接赋值（axi.AWVALID <= ...）驱动，因此它属于
    //   「为 EDA 工具（irun/VCS）保留的可选路径」，见 axi/PROTOCOL.md。
    //
    //   副作用：Verilator -Wall 会报 8 处
    //     MULTIDRIVENPROC: "written to in clocking block also written by another process"
    //   因为 clocking 的 output 声明与 TB 的直接赋值都算「驱动同一信号」。
    //   这不是真实的多驱动冲突（clocking 从未被调用），但会污染告警列表。
    //
    //   若将来决定只用 Verilator 一条路径，可安全删除本 block
    //   （实测删除后 25 个套件全部照常通过）；保留则是为了 EDA 兼容。
    clocking cb @(posedge ACLK);
        default input #1step output #1;
        // master 侧驱动
        output AWVALID, AWADDR, WVALID, WDATA, BREADY, ARVALID, ARADDR, RREADY;
        input  AWREADY, WREADY, BVALID, BRESP, ARREADY, RVALID, RDATA, RRESP;
    endclocking

    // ---- SVA 断言（协议合法性，固定延迟形式，Verilator/slang 均可编译） ----
    // VALID 稳定性：VALID=1 且 READY=0（握手未完成）时，下一拍 VALID 必须保持 1；
    //              握手完成拍（READY=1）后 VALID 可合法撤下。
    ap_aw_valid_stable:
        assert property (@(posedge ACLK) (AWVALID && !AWREADY) |=> AWVALID)
        else $error("AXI: AWVALID dropped before handshake");

    ap_w_valid_stable:
        assert property (@(posedge ACLK) (WVALID && !WREADY) |=> WVALID)
        else $error("AXI: WVALID dropped before handshake");

    ap_ar_valid_stable:
        assert property (@(posedge ACLK) (ARVALID && !ARREADY) |=> ARVALID)
        else $error("AXI: ARVALID dropped before handshake");

    ap_b_valid_stable:
        assert property (@(posedge ACLK) (BVALID && !BREADY) |=> BVALID)
        else $error("AXI: BVALID dropped before handshake");

    ap_r_valid_stable:
        assert property (@(posedge ACLK) (RVALID && !RREADY) |=> RVALID)
        else $error("AXI: RVALID dropped before handshake");

    // ---- 关于 READY 保持：不添加断言 ----
    // 曾尝试加 5 条「READY 一旦拉高必须保持」的断言，实测立刻误报
    // （ap_aw_ready_stable: AWREADY dropped before handshake）。
    // 核查 AXI 规范后确认是**断言本身写错了**：
    //   AXI 对 READY 的唯一要求是「握手完成前不得撤销」，
    //   而握手的定义是 (VALID && READY)；当 VALID=0 时并没有进行中的握手，
    //   READY 完全可以自由撤销。桥在 S_AW_H/S_IDLE 之间切换时正是这种情况。
    //   真正需要保证的「VALID 不中途撤销」已由上面 5 条断言覆盖
    //   （VALID 保持 ⟺ 握手机会不被吞掉）。
    // 结论：不加 READY 侧断言，这不是覆盖缺口。

endinterface

`endif // AXI_LITE_IF_SV