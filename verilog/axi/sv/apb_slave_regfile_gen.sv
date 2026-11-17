// ============================================================================
// apb_slave_regfile_gen：APB 寄存器从机（由 gen_regfile.py 从 regfile_spec.json 生成）
// ============================================================================
// ⚠️ 本文件为自动生成，请勿手工编辑。修改 regfile_spec.json 后重新生成。
//
// 寄存器映射：
    //   [ 0] 0x00  VERSION      RO    reset=0x00010203   IP 版本号（只读，整字属性）
    //   [ 1] 0x04  INT_STATUS   W1C   reset=0x000000F3   中断状态（整字 W1C，复位带初始待处理位）
    //   [ 2] 0x08  RESERVED0    RSVD  reset=0x00000000   保留（整字保留）
    //   [ 3] 0x0C  CTRL         RW    reset=0xA53C0000   控制寄存器（含位域定义）（位域属性见 fields）（reset 由各位域 reset 拼合得出，非 0）
    //   [ 4] 0x10  STATUS       RO    reset=0x00000011   状态寄存器（整字只读）
    //   [ 5] 0x14  INT_ENABLE   RW    reset=0x00000000   中断使能（整字读写）
    //   [ 6] 0x18  DATA_IN      RW    reset=0x00000000   数据输入
    //   [ 7] 0x1C  DATA_OUT     RW    reset=0x00000000   数据输出
    //   [ 8] 0x20  SCRATCH      RW    reset=0x00000000   暂存寄存器
    //   [ 9] 0x24  IRQ_CLEAR    W1C   reset=0x0000000F   中断清除（整字 W1C，复位带初始值）
    //   [10] 0x28  INT_SET      W1S   reset=0x00000000   中断置位（写 1 置位）
    //   [11] 0x2C  EVENT_CNT    RC    reset=0x000000A5   事件计数（读清零，复位带初始值）
//
// 寄存器属性语义：
//   RO    只读：写被忽略，读返回复位时的固定值
//   RW    读写：写什么读什么
//   W1C   写 1 清零：写 1 的位清零，写 0 的位保持
//   RSVD  保留：读恒 0，写忽略
// ============================================================================
`ifndef APB_SLAVE_REGFILE_GEN_SV
`define APB_SLAVE_REGFILE_GEN_SV
`timescale 1ns / 1ps

module apb_slave_regfile_gen #(
    parameter int AW          = 8,
    parameter int DW          = 32,
    parameter int NREG        = 12,
    parameter int WAIT_CYCLES = 2
)(
    input  logic PCLK,
    input  logic PRESETn,
    apb_if.slave apb
);

    // 参数约束：NREG 必须至少覆盖规格里的全部寄存器。
    // 规格有 12 个寄存器；若 NREG 更小，后面的寄存器整片消失，
    // 访问它们会返回 SLVERR —— 看起来像「越界保护」，实则是配置错误。
    // 实测：NREG=4 时偏移 0x10 起全部读不到（本轮踩到并修复）。
    initial begin
        if (NREG < 12) begin
            $display("ERROR: NREG=%0d 小于规格寄存器数 12（部分寄存器将不存在）",
                     NREG);
            $fatal(1, "invalid NREG parameter");
        end
        // WAIT_CYCLES 内部是 8 位计数器（wait_cnt[7:0]），>255 会被静默截断。
        // 实测 WAIT_CYCLES=256 时延迟只有 7 周期（期望 263），无任何提示。
        if (WAIT_CYCLES < 0 || WAIT_CYCLES > 255) begin
            $display("ERROR: WAIT_CYCLES=%0d 越界（有效范围 0..255，内部为 8 位计数器）",
                     WAIT_CYCLES);
            $fatal(1, "invalid WAIT_CYCLES parameter");
        end
        if (AW < 2 || AW > 32) begin
            $display("ERROR: AW=%0d 越界（有效范围 2..32）", AW);
            $fatal(1, "invalid AW parameter");
        end
        // 地址空间必须装得下寄存器堆，否则尾部寄存器不可访问。
        if ((NREG * (DW/8)) > (1 << AW)) begin
            $display("ERROR: NREG=%0d*DW/8=%0d 字节 > 地址空间 %0d 字节（AW=%0d）",
                     NREG, NREG*(DW/8), (1 << AW), AW);
            $fatal(1, "invalid NREG/AW combination");
        end
    end

    import apb_defs_pkg::*;

    // ---- 寄存器属性编码 ----
    localparam int A_RSVD = 0;
    localparam int A_RW   = 1;
    localparam int A_RO   = 2;
    localparam int A_W1C  = 3;
    localparam int A_W1S  = 4;   // 写 1 置位
    localparam int A_RC   = 5;   // 读清零（读完成拍清零）

    // ---- 寄存器名到索引（供 TB / 软件参考）----
    localparam int VERSION      = 0;   // 0x00
    localparam int INT_STATUS   = 1;   // 0x04
    localparam int RESERVED0    = 2;   // 0x08
    localparam int CTRL         = 3;   // 0x0C
    localparam int STATUS       = 4;   // 0x10
    localparam int INT_ENABLE   = 5;   // 0x14
    localparam int DATA_IN      = 6;   // 0x18
    localparam int DATA_OUT     = 7;   // 0x1C
    localparam int SCRATCH      = 8;   // 0x20
    localparam int IRQ_CLEAR    = 9;   // 0x24
    localparam int INT_SET      = 10;   // 0x28
    localparam int EVENT_CNT    = 11;   // 0x2C

    // ---- 属性查询（按索引）----
    function automatic int attr_of(input int i);
        case (i)
            0: return A_RO;   // VERSION @0x00
            1: return A_W1C;   // INT_STATUS @0x04
            2: return A_RSVD;   // RESERVED0 @0x08
            3: return A_RW;   // CTRL @0x0C
            4: return A_RO;   // STATUS @0x10
            5: return A_RW;   // INT_ENABLE @0x14
            6: return A_RW;   // DATA_IN @0x18
            7: return A_RW;   // DATA_OUT @0x1C
            8: return A_RW;   // SCRATCH @0x20
            9: return A_W1C;   // IRQ_CLEAR @0x24
            10: return A_W1S;   // INT_SET @0x28
            11: return A_RC;   // EVENT_CNT @0x2C
            default: return A_RSVD;   // 未定义地址视为保留
        endcase
    endfunction

    // ---- 复位值查询（按索引）----
    function automatic logic [DW-1:0] reset_of(input int i);
        case (i)
            0: return 32'h00010203;
            1: return 32'h000000F3;
            2: return 32'h00000000;
            3: return 32'hA53C0000;
            4: return 32'h00000011;
            5: return 32'h00000000;
            6: return 32'h00000000;
            7: return 32'h00000000;
            8: return 32'h00000000;
            9: return 32'h0000000F;
            10: return 32'h00000000;
            11: return 32'h000000A5;
            default: return '0;
        endcase
    endfunction

    // ---- 寄存器堆 ----
    logic [DW-1:0] mem [0:NREG-1];

    // ---- 地址译码 ----
    logic [AW-1:0] idx;
    logic          in_range;
    assign idx = apb.PADDR >> 2;
    assign in_range = (idx < NREG);

    // CTRL 位域写变换（逐位域按属性合并）
    function automatic logic [DW-1:0] field_write_CTRL(
        input logic [DW-1:0] old_v,
        input logic [DW-1:0] wr_v
    );
        logic [DW-1:0] nv;
        nv = old_v;
        // ENABLE bits[0:0] RW
        nv = (nv & ~32'h00000001) | (wr_v & 32'h00000001);
        // MODE bits[2:1] RW
        nv = (nv & ~32'h00000006) | (wr_v & 32'h00000006);
        // IRQ_EN bits[3:3] W1S
        nv = (nv & ~32'h00000008) | ((old_v | wr_v) & 32'h00000008);
        // RESERVED bits[15:4] RSVD
        nv = (nv & ~32'h0000FFF0) | (old_v & 32'h0000FFF0);
        // STATUS_CODE bits[23:16] RC
        nv = (nv & ~32'h00FF0000) | (old_v & 32'h00FF0000);
        // FLAGS bits[31:24] W1C
        nv = (nv & ~32'hFF000000) | (old_v & ~wr_v & 32'hFF000000);
        return nv;
    endfunction

    // ---- 位域寄存器写分派（按索引）----
    function automatic logic [DW-1:0] write_dispatch(
        input int            idx,
        input logic [DW-1:0] old_v,
        input logic [DW-1:0] wr_v
    );
        case (idx)
            3: return field_write_CTRL(old_v, wr_v);
            default: return write_transform_word(old_v, wr_v, attr_of(idx));
        endcase
    endfunction

    // CTRL 位域读清零（仅清 RC 位域）
    function automatic logic [DW-1:0] field_read_clear_CTRL(
        input logic [DW-1:0] cur_v
    );
        logic [DW-1:0] nv;
        nv = cur_v;
        nv = nv & ~32'h00FF0000;   // STATUS_CODE [23:16] RC 读清零
        return nv;
    endfunction

    // ---- 位域寄存器读清零分派 ----
    function automatic logic [DW-1:0] read_clear_dispatch_fld(
        input int            idx,
        input logic [DW-1:0] cur_v
    );
        case (idx)
            3: return field_read_clear_CTRL(cur_v);
            default: return read_clear_word(cur_v, attr_of(idx));
        endcase
    endfunction


    // ---- 写数据按属性变换（整字寄存器）----
    function automatic logic [DW-1:0] write_transform_word(
        input logic [DW-1:0] old_v,
        input logic [DW-1:0] wr_v,
        input int            attr
    );
        case (attr)
            A_RW:  return wr_v;
            A_W1C: return old_v & ~wr_v;
            A_W1S: return old_v | wr_v;   // 写 1 置位
            default: return old_v;        // RO / RC / RSVD 忽略写
        endcase
    endfunction

    // ---- 读数据按属性变换 ----
    function automatic logic [DW-1:0] read_transform(
        input logic [DW-1:0] cur_v,
        input int            attr
    );
        return (attr == A_RSVD) ? '0 : cur_v;
    endfunction

    // ---- 读清零（RC）：读完成拍清除 ----
    //   整字 RC：读后整体清 0
    //   位域 RC：只清 RC 位域，其他位域保持不变
    function automatic logic [DW-1:0] read_clear_word(
        input logic [DW-1:0] cur_v,
        input int            attr
    );
        return (attr == A_RC) ? '0 : cur_v;
    endfunction

    // read_clear_dispatch_fld 由下方位域代码块提供（若有位域寄存器）

    // ---- 阶段状态机 ----
    apb_phase_t phase;
    logic [7:0] wait_cnt;

    always_ff @(posedge PCLK or negedge PRESETn) begin
        if (!PRESETn) begin
            phase       <= APB_IDLE;
            wait_cnt    <= '0;
            apb.PREADY  <= 1'b0;
            apb.PSLVERR <= 1'b0;
            apb.PRDATA  <= '0;
            for (int i = 0; i < NREG; i++)
                mem[i] <= reset_of(i);
        end else begin
            case (phase)
                APB_IDLE: begin
                    apb.PREADY  <= 1'b0;
                    apb.PSLVERR <= 1'b0;
                    if (apb.PSEL && !apb.PENABLE) begin
                        wait_cnt <= WAIT_CYCLES[7:0];
                        phase    <= APB_SETUP;
                    end
                end
                APB_SETUP: begin
                    apb.PREADY  <= 1'b0;
                    apb.PSLVERR <= 1'b0;
                    if (apb.PENABLE)
                        phase <= APB_ACC;
                end
                APB_ACC: begin
                    if (wait_cnt == 0) begin
                        apb.PREADY  <= 1'b1;
                        apb.PSLVERR <= in_range ? 1'b0 : 1'b1;
                        if (in_range && apb.PWRITE)
                            mem[idx] <= write_dispatch(idx, mem[idx], apb.PWDATA);
                        if (!apb.PWRITE) begin
                            apb.PRDATA <= in_range ? read_transform(mem[idx], attr_of(idx)) : '0;
                            // RC（读清零）：读完成拍清除（位域级）
                            if (in_range)
                                mem[idx] <= read_clear_dispatch_fld(idx, mem[idx]);
                        end
                        phase <= APB_IDLE;
                    end else begin
                        apb.PREADY  <= 1'b0;
                        wait_cnt    <= wait_cnt - 1'b1;
                    end
                end
                default: begin
                    phase       <= APB_IDLE;
                    apb.PREADY  <= 1'b0;
                    apb.PSLVERR <= 1'b0;
                end
            endcase
        end
    end

endmodule
`endif // APB_SLAVE_REGFILE_GEN_SV
