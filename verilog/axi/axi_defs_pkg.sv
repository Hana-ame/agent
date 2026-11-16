// ============================================================================
// axi_defs_pkg：AXI4 / AXI4-Lite 协议的完整定义层
// ============================================================================
`timescale 1ns / 1ps
// 把协议的知识集中成"单一事实来源"，供转换设计（桥）与 testbench 共同引用：
//   - AXI 五通道的 struct 打包定义（地址/数据/响应通道全部字段）
//   - 协议常量：响应编码、突发类型、大小编码、Cache/Prot 常用值
//   - 协议函数：响应名称、地址对齐、单拍位选（DUT 与参考模型共用同一逻辑）
//
// 本文件提供**完整的 AXI4 五通道定义**（不是 Lite 子集）：
//   AW: 写地址通道   { AWID, AWADDR, AWLEN, AWSIZE, AWBURST, AWLOCK,
//                      AWCACHE, AWPROT, AWQOS, AWREGION }
//   W : 写数据通道   { WDATA, WSTRB, WLAST }
//   B : 写响应通道   { BID, BRESP }
//   AR: 读地址通道   { ARID, ARADDR, ARLEN, ARSIZE, ARBURST, ARLOCK,
//                      ARCACHE, ARPROT, ARQOS, ARREGION }
//   R : 读数据通道   { RID, RDATA, RRESP(=RRESP), RLAST }
// 全局信号：ACLK, ARESETn
//
// 说明：本桥只用到 AXI4-Lite 子集（无 burst、单拍传输）。
//       struct 保留 AXI4 全量字段，用途是「协议对照与日后扩展」——
//       ⚠ 这些 AXI4 专有字段（awlen/awsize/awburst/awcache/awprot/
//         awqos/awregion/wstrb/wlast/bid/arid/rlast 等）目前
//         **未被任何 RTL 或 TB 引用**（grep 结果为 0），
//         不是「恒为默认值」，而是「根本未使用」。
//       桥与接口的信号集就是 AXI4-Lite 的 16 个信号，
//       见 sv/axi_lite_if.sv。若日后要支持 burst，需同时扩展接口与桥。
//
// 工具链：Verilator 5.055（仿真/覆盖率）+ slang 12.1（编译检查）。
//         iverilog 已从工具链移除（见 README）。
// ============================================================================
`ifndef AXI_DEFS_PKG_SV
`define AXI_DEFS_PKG_SV

package axi_defs_pkg;

    // ------------------------------------------------------------------
    // 1. 全局信号 / 位宽参数（AXI4-Lite 默认）
    // ------------------------------------------------------------------
    parameter int PW     = 8;     // 地址位宽（本桥环境）
    parameter int DW     = 32;    // 数据位宽
    parameter int IDW    = 0;     // ID 位宽（AXI4-Lite 通常无 ID，设 0 表示简化）
    parameter int PROT_W = 3;     // 保护类型位宽

    // ------------------------------------------------------------------
    // 2. 响应编码（RESP / BRESP / RRESP 共用）
    // ------------------------------------------------------------------
    parameter logic [1:0] RESP_OKAY   = 2'b00;  // 正常访问成功
    parameter logic [1:0] RESP_EXOKAY = 2'b01;  // 独占访问成功
    parameter logic [1:0] RESP_SLVERR = 2'b10;  // 从机错误（访问失败）
    parameter logic [1:0] RESP_DECERR = 2'b11;  // 解码错误（地址无主）

    // ------------------------------------------------------------------
    // 3. 突发类型（AXI4；AXI4-Lite 不使用突发，但定义完整 AXI 所需）
    // ------------------------------------------------------------------
    parameter logic [1:0] BURST_FIXED = 2'b00;  // 固定地址突发
    parameter logic [1:0] BURST_INCR  = 2'b01;  // 递增突发
    parameter logic [1:0] BURST_WRAP  = 2'b10;  // 回卷突发

    // ------------------------------------------------------------------
    // 4. 大小编码（AxSIZE：字节数 = 2^AxSIZE）
    // ------------------------------------------------------------------
    parameter int SIZE_BYTE = 3'd0;
    parameter int SIZE_HALF = 3'd1;
    parameter int SIZE_WORD = 3'd2;
    parameter int SIZE_DWRD = 3'd3;

    // ------------------------------------------------------------------
    // 5. 锁 / 缓存 / 保护 常用值
    // ------------------------------------------------------------------
    parameter logic       LOCK_NORMAL  = 1'b0;  // 普通访问
    parameter logic       LOCK_EXCL    = 1'b1;  // 独占访问
    parameter logic [3:0] CACHE_DEF    = 4'b0011; // 可缓存、可缓冲（默认）
    parameter logic [2:0] PROT_UNPRIV  = 3'b000;  // 非特权、安全、数据访问

    // ------------------------------------------------------------------
    // 6. 五通道 struct（打包，全部字段齐全，形成"完整 AXI 定义"）
    // ------------------------------------------------------------------
    // 6.1 写地址通道 AW
    typedef struct packed {
        logic [IDW-1:0]  awid;     // 写事务 ID
        logic [PW-1:0]   awaddr;   // 写地址
        logic [7:0]      awlen;    // 突发长度（AXI4-Lite 恒 0）
        logic [2:0]      awsize;   // 突发大小
        logic [1:0]      awburst;  // 突发类型
        logic            awlock;   // 锁类型
        logic [3:0]      awcache;  // 缓存类型
        logic [PROT_W-1:0] awprot; // 保护类型
        logic [3:0]      awqos;    // 服务质量（AXI4；Lite 保留）
        logic [3:0]      awregion; // 区域标识（AXI4；Lite 保留）
    } aw_chan_t;

    // 6.2 写数据通道 W
    typedef struct packed {
        logic [DW-1:0]   wdata;    // 写数据
        logic [DW/8-1:0] wstrb;    // 写 strobe（字节使能）
        logic            wlast;    // 最后一笔数据
    } w_chan_t;

    // 6.3 写响应通道 B
    typedef struct packed {
        logic [IDW-1:0]  bid;      // 响应 ID
        logic [1:0]      bresp;    // 响应编码
    } b_chan_t;

    // 6.4 读地址通道 AR
    typedef struct packed {
        logic [IDW-1:0]  arid;     // 读事务 ID
        logic [PW-1:0]   araddr;   // 读地址
        logic [7:0]      arlen;    // 突发长度
        logic [2:0]      arsize;   // 突发大小
        logic [1:0]      arburst;  // 突发类型
        logic            arlock;   // 锁类型
        logic [3:0]      arcache;  // 缓存类型
        logic [PROT_W-1:0] arprot; // 保护类型
        logic [3:0]      arqos;    // 服务质量
        logic [3:0]      arregion; // 区域标识
    } ar_chan_t;

    // 6.5 读数据通道 R
    typedef struct packed {
        logic [IDW-1:0]  rid;      // 响应 ID
        logic [DW-1:0]   rdata;    // 读数据
        logic [1:0]      rresp;    // 响应编码
        logic            rlast;    // 最后一笔数据
    } r_chan_t;

    // ------------------------------------------------------------------
    // 7. 协议函数（供 DUT 与参考模型共用，保证"唯一事实来源"）
    // ------------------------------------------------------------------
    // 响应编码 -> 助记名
    function automatic string resp_name(input logic [1:0] r);
        case (r)
            RESP_OKAY:   return "OKAY";
            RESP_EXOKAY: return "EXOKAY";
            RESP_SLVERR: return "SLVERR";
            default:     return "DECERR";
        endcase
    endfunction

    // 地址对齐掩码：按 SIZE 返回字节偏移应全零的位数
    // 例: SIZE=WORD(2) -> 低 2 位须为 0（4 字节对齐）
    // 地址对齐判断（协议工具函数）。
    // 注意：AXI4-Lite **不要求**地址对齐（协议只「建议」），本桥与从机
    // 也不强制对齐 —— 非对齐/越界地址由从机用 SLVERR 处理。
    // 因此该函数不用于监视器检查（曾试过，会对 gaps 套件的越界测试大量误报），
    // 保留供使用者按需调用（如验证「对齐访问」的专项测试）。
    function automatic logic is_aligned(input logic [PW-1:0] addr,
                                        input logic [2:0]   size);
        logic [7:0] mask;
        mask = (8'h1 << size) - 8'h1;   // 2^size - 1
        return ((addr & mask[PW-1:0]) == {PW{1'b0}});
    endfunction

    // 单事务桥的通道合法性：AW 与 AR 不允许同时请求
    // （并发桥读写并行，不适用此规则 —— 由监视器的 SERIAL_BRIDGE 参数隔离）
    function automatic logic valid_channels_ok(input logic awv, input logic wv,
                                               input logic arv);
        // 单事务桥：AW 与 AR 不允许同时请求（否则需要仲裁/流水）
        return !(awv && arv);
    endfunction

endpackage

`endif // AXI_DEFS_PKG_SV