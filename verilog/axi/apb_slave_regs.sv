// APB slave：慢速寄存器堆外设（带 wait states + 地址越界 SLVERR）
//
// 协议要点（APB v2.0）：
//   - 事务分 SETUP（PSEL=1, PENABLE=0）和 ACCESS（PSEL=1, PENABLE=1）两阶段
//   - ACCESS 阶段 PREADY=0 时插入 wait state（"可以慢"）
//   - PREADY=1 且 PSEL=1 且 PENABLE=1 时事务完成
//     （读：PRDATA 有效；写：采样写入；PSLVERR 可选置位表示错误）
//   - 地址越界（>= NREG*4）返回 PSLVERR=1，不写存储器
`timescale 1ns / 1ps

module apb_slave_regs #(
    parameter AW          = 8,   // 地址位宽
    parameter DW          = 32,  // 数据位宽
    parameter NREG        = 16,  // 寄存器个数（地址 0..NREG*4-4）
    parameter WAIT_CYCLES = 2    // ACCESS 阶段插入的 wait states 数
)(
    input  wire             PCLK,
    input  wire             PRESETn,
    input  wire             PSEL,
    input  wire             PENABLE,
    input  wire             PWRITE,
    input  wire [AW-1:0]    PADDR,
    input  wire [DW-1:0]    PWDATA,
    output reg              PREADY,
    output reg  [DW-1:0]    PRDATA,
    output reg              PSLVERR
);

    // ---- 寄存器堆（写使能由 PWRITE 在完成拍控制） ----
    reg [DW-1:0] mem [0:NREG-1];

    integer i;
    // 上电清零（保证仿真从确定状态开始）
    initial begin
        for (i = 0; i < NREG; i = i + 1) mem[i] = {DW{1'b0}};
    end

    // ---- 地址译码 ----
    wire [AW-1:0] idx   = PADDR >> 2;              // 字序号
    wire          in_range = (idx < NREG);

    // ---- ACCESS 阶段 wait state 计数 ----
    reg [7:0] wait_cnt;
    reg       in_access;   // 当前处于 ACCESS 阶段

    always @(posedge PCLK or negedge PRESETn) begin
        if (!PRESETn) begin
            in_access <= 1'b0;
            wait_cnt  <= 8'h0;
            PREADY    <= 1'b0;
            PSLVERR   <= 1'b0;
            PRDATA    <= {DW{1'b0}};
        end else begin
            // 默认：ACCESS 结束（或未开始）时 PREADY 复位
            if (PSEL && PENABLE) begin
                // ---- ACCESS 阶段 ----
                if (wait_cnt == 0) begin
                    // wait 耗尽：本拍完成事务（PREADY 拉高 1 拍）
                    PREADY  <= 1'b1;
                    PSLVERR <= in_range ? 1'b0 : 1'b1;   // 越界报错
                    if (in_range && PWRITE)
                        mem[idx] <= PWDATA;              // 时序写：完成拍锁存
                    if (!PWRITE)
                        PRDATA <= in_range ? mem[idx] : {DW{1'b0}};
                    in_access <= 1'b0;                   // 下一拍离开 ACCESS
                end else begin
                    // 插入 wait state：PREADY 拉低，计数递减
                    PREADY   <= 1'b0;
                    wait_cnt <= wait_cnt - 1'b1;
                    in_access <= 1'b1;
                end
            end else begin
                // ---- SETUP 或 IDLE ----
                if (PSEL && !PENABLE) begin
                    // SETUP 阶段：准备 wait 计数器（合法：PREADY 保持 0）
                    wait_cnt  <= WAIT_CYCLES[7:0];
                    in_access <= 1'b0;
                end
                PREADY  <= 1'b0;
                PSLVERR <= 1'b0;
            end
        end
    end

endmodule