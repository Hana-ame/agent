// ============================================================================
// apb_defs_pkg：APB（AMBA 3/4 APB）协议的完整定义层
// ============================================================================
`timescale 1ns / 1ps
// 单一事实来源：把 APB 总线信号、阶段状态、时序规则集中定义，
// 供 APB 从机（dut）、AXI-APB 桥（转换设计）与 testbench 共同引用。
//
// APB 总线信号（全局）：
//   PCLK    时钟（所有传输与 PCLK 沿同步）
//   PRESETn 复位（低有效）
//   PSELx   从机选择（每从机一个）
//   PENABLE 访问使能（ACCESS 阶段标志）
//   PWRITE  传输方向（1=写, 0=读）
//   PADDR   地址
//   PWDATA  写数据
//   PRDATA  读数据
//   PREADY  从机就绪（慢速从机用 0 插入 wait states -> "可以慢"）
//   PSLVERR 传输错误（APB3 引入；可选）
//
// 本实现对应的 APB 版本：**APB3**
//   已具备 PSLVERR（APB3 新增）。
//   ⚠ 未实现 APB4 的两个信号：PPROT（保护属性）、PSTRB（写字节选通）。
//     因此本接口是 APB3 信号集，不支持「按字节写」与「安全/特权标记」。
//     如需 APB4，需在 sv/apb_if.sv 增加 PPROT/PSTRB，
//     并让从机的写逻辑按 PSTRB 做字节使能。
//
// 阶段（Transfer）：
//   IDLE    无传输（PSEL=0, PENABLE=0）
//   SETUP   建立（PSEL=1, PENABLE=0）：地址/方向/PWDATA 在沿前有效，
//           至少维持 1 拍；期间 PREADY 不得为 1
//   ACCESS  访问（PSEL=1, PENABLE=1）：数据相位；PREADY=1 且保持到
//           本拍结束才完成；PREADY=0 时 PENABLE 必须保持（wait state）
//
// 规则（"可以慢，但一定要合法"）：
//   R1: PENABLE 只能在 PSEL=1 时拉高（否则非法）
//   R2: ACCESS 一旦进入，若 PREADY=0 不得中途退出
//   R3: SETUP 期间从机不拉高 PREADY
//   R4: 传输在「PREADY=1 && PENABLE=1 && PSEL=1」完成
//   R5: 完成沿之后，PENABLE 与 PSEL 都应回到 0（下一传输重新 SETUP）
//   R6: 写数据在完成沿锁存；读数据在完成沿后有效
//   R7: PSLVERR 只在完成沿有效（报告本次传输错误）
// ============================================================================
`ifndef APB_DEFS_PKG_SV
`define APB_DEFS_PKG_SV

package apb_defs_pkg;

    // ------------------------------------------------------------------
    // 1. 位宽参数
    // ------------------------------------------------------------------
    parameter int AW = 8;
    parameter int DW = 32;

    // ------------------------------------------------------------------
    // 2. APB 总线信号（打包 struct：完整 APB 定义）
    //    注意：PCLK/PRESETn 是全局时钟/复位，不放入事务 struct
    // ------------------------------------------------------------------
    typedef struct packed {
        logic            psel;     // 从机选择
        logic            penable;  // 访问使能（ACCESS 阶段标志）
        logic            pwrite;   // 写/读方向
        logic [AW-1:0]   paddr;    // 地址
        logic [DW-1:0]   pwdata;   // 写数据
        logic            pready;   // 从机就绪
        logic [DW-1:0]   prdata;   // 读数据
        logic            pslverr;  // 传输错误
    } apb_bus_t;

    // ------------------------------------------------------------------
    // 3. 阶段状态（状态机类型）
    // ------------------------------------------------------------------
    typedef enum logic [1:0] {
        APB_IDLE  = 2'b00,   // 无传输
        APB_SETUP = 2'b01,   // 建立阶段
        APB_ACC   = 2'b10    // 访问阶段
    } apb_phase_t;

    // ------------------------------------------------------------------
    // 4. 时序规则函数（"合法"的形式化检查；供 monitor 调用）
    // ------------------------------------------------------------------
    // R1: PENABLE 只能在 PSEL=1 时拉高
    function automatic logic penable_ok(input logic psel, input logic penable);
        return !(penable && !psel);
    endfunction

    // R2/R3: SETUP 阶段不拉高 PREADY；ACCESS 未完成前不退出
    //  fold: 组合判定当前 APB 状态是否"合法"（在给定前后拍信息下）
    //  prev_access, prev_ready 由调用方寄存
    function automatic logic phase_transition_ok(
        input logic prev_access, input logic prev_ready,
        input logic cur_access);
        // ACCESS(PREADY=0) -> 非 ACCESS 即非法
        if (prev_access && !prev_ready && !cur_access)
            return 1'b0;
        return 1'b1;
    endfunction

    // R4: 传输完成条件
    function automatic logic transfer_done(input logic psel, input logic penable,
                                           input logic pready);
        return psel && penable && pready;
    endfunction

    // 阶段名（调试打印用）
    function automatic string phase_name(input apb_phase_t p);
        case (p)
            APB_IDLE:  return "IDLE";
            APB_SETUP: return "SETUP";
            APB_ACC:   return "ACCESS";
            default:   return "?";
        endcase
    endfunction

    // ------------------------------------------------------------------
    // 5. 便捷构造函数：干净的总线默认值
    // ------------------------------------------------------------------
    function automatic apb_bus_t apb_idle_bus();
        apb_bus_t b;
        b = '0;   // 全零 = 无选择、无使能、无等待
        return b;
    endfunction

endpackage

`endif // APB_DEFS_PKG_SV