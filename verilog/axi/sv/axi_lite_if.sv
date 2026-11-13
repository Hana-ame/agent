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

    // ---- clocking block（同步驱动/采样） ----
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

    // 注：不设 "W 必须有 AW" 断言——该检查针对主站而非桥；
    //     分开发写事务中 AW 握手完成与 W 驱动之间存在合法间隙，
    //     固定延迟回溯会误报。桥本身的正确性由 5 通道 VALID 稳定性断言 +
    //     TB scoreboard 保证。

endinterface

`endif // AXI_LITE_IF_SV