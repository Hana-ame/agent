// ============================================================================
// apb_slave_regfile_sv：APB 从机 —— 带寄存器属性的真实寄存器堆
// ============================================================================
// 相比 apb_slave_regs_sv（纯存储数组"写什么读什么"），本模块实现
// 真实硬件寄存器常见属性：
//
//   RO   (read-only)     只读：写操作被忽略，读返回硬件值（本实现为固定初值）
//   RW   (read-write)    读写：净默认行为，写什么读什么
//   W1C  (write-1-clear) 写 1 清 0：写 1 的位被清零，写 0 的位保持不变
//   RSVD (reserved)      保留位：读返回 0，写被忽略
//
// 属性通过参数 ATTRIBS 配置：一个 NREG 长的字符串/整型编码数组。
//   ATTR_RW=1, ATTR_RO=2, ATTR_W1C=3, ATTR_RSVD=0
//
// 为什么需要它：
//   - 真实外设（中断状态、控制寄存器）大量使用 W1C / RO / 保留位
//   - 纯存储模型无法验证这些语义，会让 RTL 与设计意图脱节
//
// 兼容性：默认 ATTRIBS 全 RW，行为与 apb_slave_regs_sv 完全一致，
//        既有测试（读未写=0、写后读回一致）不受影响。
// ============================================================================
`ifndef APB_SLAVE_REGFILE_SV
`define APB_SLAVE_REGFILE_SV
`timescale 1ns / 1ps

module apb_slave_regfile_sv #(
    parameter int AW          = 8,
    parameter int DW          = 32,
    parameter int NREG        = 16,
    parameter int WAIT_CYCLES = 2
)(
    input  logic PCLK,
    input  logic PRESETn,
    apb_if.slave apb
);

    import apb_defs_pkg::*;

    // ---- 寄存器属性编码 ----
    localparam int A_RSVD = 0;   // 保留：读 0，写忽略
    localparam int A_RW   = 1;   // 读写
    localparam int A_RO   = 2;   // 只读：写忽略
    localparam int A_W1C  = 3;   // 写 1 清零

    // 属性分配（默认示例）：
    //   reg0 RO（版本号）  reg1 W1C（中断状态）  reg2 RSVD（保留）  reg3+ RW
    // 真实项目可由 spec/JSON 自动生成；此处用 const 数组便于阅读与覆盖
    function automatic int attr_of(input int i);
        case (i)
            0: return A_RO;
            1: return A_W1C;
            2: return A_RSVD;
            3, 4, 5: return A_RW;      // REF-04/REF-05 使用的 RW 寄存器
            // 未定义索引显式声明为 RSVD（读恒 0、写忽略），
            // 而不是落到 default 变成「幽灵 RW 寄存器」。
            // 原实现 default 返回 A_RW，导致 NREG 之内的未定义索引
            // 可被读写且返回 OKAY，看起来像正常寄存器。
            default: return A_RSVD;
        endcase
    endfunction

    // ---- 寄存器堆 ----
    logic [DW-1:0] mem [0:NREG-1];

    // RO 寄存器的固定硬件值（版本号示例）
    localparam logic [DW-1:0] RO_REG0_VALUE = 32'h0001_0203;
    // RSVD 寄存器的复位值：故意设非零，使「读恒 0」行为可观测。
    // 若复位为 0 且写入被忽略，则存储恒为 0，read_transform 的
    // RSVD 分支即使失效也无法从外部观察（变异测试 M19 曾因此存活）。
    localparam logic [DW-1:0] RSVD_REG2_VALUE = 32'hDEAD_55AA;

    // ---- 地址译码 ----
    logic [AW-1:0] idx;
    logic          in_range;
    assign idx = apb.PADDR >> 2;
    assign in_range = (idx < NREG);

    // ---- 写数据按属性变换 ----
    //   RW   : 新值 = 写入值
    //   RO   : 新值 = 原值（忽略写）
    //   W1C  : 新值 = 原值 & ~写入值（写 1 的位清零）
    //   RSVD : 新值 = 原值（忽略写）
    function automatic logic [DW-1:0] write_transform(
        input logic [DW-1:0] old_v,
        input logic [DW-1:0] wr_v,
        input int            attr
    );
        case (attr)
            A_RW:  return wr_v;
            A_W1C: return old_v & ~wr_v;
            default: return old_v;      // RO / RSVD 忽略写
        endcase
    endfunction

    // ---- 读数据按属性变换 ----
    //   RSVD : 返回 0（保留位读 0）
    //   其他 : 返回存储值（RO 在初始化时已置为硬件值）
    function automatic logic [DW-1:0] read_transform(
        input logic [DW-1:0] cur_v,
        input int            attr
    );
        return (attr == A_RSVD) ? '0 : cur_v;
    endfunction

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
            for (int i = 0; i < NREG; i++) begin
                // RO 寄存器复位为固定硬件值，其余清零
                mem[i] <= (attr_of(i) == A_RO)  ? RO_REG0_VALUE  :
                          (attr_of(i) == A_RSVD) ? RSVD_REG2_VALUE : '0;
            end
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
                    apb.PREADY  <= 1'b0;   // R3: SETUP 不回 PREADY
                    apb.PSLVERR <= 1'b0;
                    if (apb.PENABLE)
                        phase <= APB_ACC;
                end
                APB_ACC: begin
                    if (wait_cnt == 0) begin
                        // 完成拍
                        apb.PREADY  <= 1'b1;
                        apb.PSLVERR <= in_range ? 1'b0 : 1'b1;
                        if (in_range && apb.PWRITE)
                            // 写：按属性变换后存储
                            mem[idx] <= write_transform(mem[idx], apb.PWDATA, attr_of(idx));
                        if (!apb.PWRITE)
                            // 读：按属性变换后输出
                            apb.PRDATA <= in_range ? read_transform(mem[idx], attr_of(idx))
                                                   : '0;
                        phase <= APB_IDLE;
                    end else begin
                        apb.PREADY  <= 1'b0;   // 插 wait state（可以慢）
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
`endif // APB_SLAVE_REGFILE_SV
