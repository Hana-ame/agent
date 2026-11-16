// APB 慢速从机（完整 SV 接口端口版）
// 寄存器堆 + 可配 wait states（ACCESS 阶段插入，实现"可以慢"）
// + 地址越界 SLVERR（不写存储）
// 状态机用 apb_defs_pkg::apb_phase_t 枚举，阶段定义与规则来自定义层
`ifndef APBSV_SLAVE_REGS_SV
`define APBSV_SLAVE_REGS_SV
`timescale 1ns / 1ps

module apb_slave_regs_sv #(
    parameter AW          = 8,
    parameter DW          = 32,
    parameter NREG        = 16,
    parameter WAIT_CYCLES = 2
)(
    input  logic PCLK,
    input  logic PRESETn,
    apb_if.slave apb
);

    import apb_defs_pkg::*;

    // ---- 寄存器堆 ----
    logic [DW-1:0] mem [0:NREG-1];

    // ---- 地址译码 ----
    logic [AW-1:0] idx;
    logic          in_range;
    // 地址译码：按字寻址（PADDR>>2），越界返回 SLVERR。
    // ⚠ 参数约束：地址空间必须装得下所有寄存器，即 2^AW >= NREG*4。
    //   若 2^AW < NREG*4（如 AW=1 时只有 2 字节，连一个 32 位寄存器都放不下），
    //   高位地址会被截断并回绕到同一个寄存器 —— 看似能读写，实则配置非法。
    //   tb_param_sv 的 P-02 会显式检出这种非法组合。
    assign idx = apb.PADDR >> 2;
    assign in_range = (idx < NREG);

    // ---- 阶段状态机 ----
    apb_phase_t phase;
    logic [7:0] wait_cnt;

    always_ff @(posedge PCLK or negedge PRESETn) begin
        if (!PRESETn) begin
            phase    <= APB_IDLE;
            wait_cnt <= '0;
            apb.PREADY  <= 1'b0;
            apb.PSLVERR <= 1'b0;
            apb.PRDATA  <= '0;
            for (int i = 0; i < NREG; i++) mem[i] <= '0;   // 复位清零（与写同过程）
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
                        // 完成拍：PREADY=1 + 数据有效
                        apb.PREADY  <= 1'b1;
                        apb.PSLVERR <= in_range ? 1'b0 : 1'b1;
                        if (in_range && apb.PWRITE)
                            mem[idx] <= apb.PWDATA;
                        if (!apb.PWRITE)
                            apb.PRDATA <= in_range ? mem[idx] : '0;
                        phase <= APB_IDLE;
                    end else begin
                        apb.PREADY  <= 1'b0;   // 插 wait state
                        wait_cnt    <= wait_cnt - 1'b1;
                    end
                end
                default: begin
                    phase    <= APB_IDLE;
                    apb.PREADY  <= 1'b0;
                    apb.PSLVERR <= 1'b0;
                end
            endcase
        end
    end

endmodule
`endif // APBSV_SLAVE_REGS_SV