// APB 接口（完整 SV 版）
// 供完整 SV 链路使用。iverilog 不可编译，需 slang 检查或 irun/VCS 运行。
//
// 包含：
//   - APB v2.0 总线信号（PSEL/PENABLE/PWRITE/PADDR/PWDATA/PRDATA/PREADY/PSLVERR）
//   - modport: master / slave
//   - clocking: cb
//   - SVA 断言: 阶段时序规则（R1/R2/R3），即"可以慢，但一定要合法"
//
// 规则（与 apb_defs_pkg 一致，这里用 SVA 表达）：
//   R1: PENABLE 只能在 PSEL=1 时拉高
//   R2: ACCESS 阶段 PREADY=0 时 PENABLE 必须保持（不得中途退出）
//   R3: SETUP 阶段（PSEL && !PENABLE）从机不得回 PREADY=1
//   R4: 传输在 PSEL && PENABLE && PREADY 完成（完成拍）
`ifndef APB_IF_SV
`define APB_IF_SV

interface apb_if #(
    parameter AW = 8,
    parameter DW = 32
) (input logic PCLK);

    // ---- APB 总线信号 ----
    logic        PSEL;
    logic        PENABLE;
    logic        PWRITE;
    logic [AW-1:0] PADDR;
    logic [DW-1:0] PWDATA;
    logic [DW-1:0] PRDATA;
    logic        PREADY;
    logic        PSLVERR;

    // ---- modport ----
    modport master (
        output PSEL, PENABLE, PWRITE, PADDR, PWDATA,
        input  PRDATA, PREADY, PSLVERR
    );
    modport slave (
        input  PSEL, PENABLE, PWRITE, PADDR, PWDATA,
        output PRDATA, PREADY, PSLVERR
    );
    // 监视器：只读全部信号
    modport monitor (
        input PSEL, PENABLE, PWRITE, PADDR, PWDATA,
              PRDATA, PREADY, PSLVERR
    );

    // ---- clocking block ----
    clocking cb @(posedge PCLK);
        default input #1step output #1;
        output PSEL, PENABLE, PWRITE, PADDR, PWDATA;
        input  PRDATA, PREADY, PSLVERR;
    endclocking

    // ---- SVA 断言 ----
    // R1: PENABLE 只能在 PSEL=1 时拉高
    ap_penable_requires_psel:
        assert property (@(posedge PCLK) PENABLE |-> PSEL)
        else $error("APB R1: PENABLE without PSEL");

    // R3: SETUP 阶段（PSEL && !PENABLE）从机不得回 PREADY=1
    ap_no_ready_in_setup:
        assert property (@(posedge PCLK) (PSEL && !PENABLE) |-> !PREADY)
        else $error("APB R3: PREADY during SETUP");

    // R2: ACCESS 阶段未完成（PREADY=0）时，下一拍不能退出 ACCESS
    //     （即：PENABLE 一旦拉高且 PREADY=0，必须保持到 PREADY=1）
    ap_access_stable_until_ready:
        assert property (@(posedge PCLK)
            (PSEL && PENABLE && !PREADY) |=> (PSEL && PENABLE))
        else $error("APB R2: ACCESS exited before PREADY");

    // R4(log only): 传输完成拍
    // (用 cover property 记录，便于覆盖率统计)
    cov_xfer_done: cover property (@(posedge PCLK)
        PSEL && PENABLE && PREADY);

    // R6: PSLVERR 只在「完成拍」有效
    //    APB 规范：PSLVERR 仅在 PSEL=1 && PENABLE=1 && PREADY=1 那拍有意义，
    //    其它时刻必须为 0（否则 master 可能误判一个未完成的传输为失败）。
    ap_slverr_only_on_complete:
        assert property (@(posedge PCLK)
            PSLVERR |-> (PSEL && PENABLE && PREADY))
        else $error("APB R6: PSLVERR asserted outside the completion cycle");

    // 完成拍后下一拍必须退出 ACCESS（R5: 完成后回 IDLE/SETUP）
    ap_xfer_done_exits:
        assert property (@(posedge PCLK)
            (PSEL && PENABLE && PREADY) |=> (!PENABLE))
        else $error("APB R5: PREADY high but ACCESS not exited");

endinterface

`endif // APB_IF_SV