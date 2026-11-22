// 8-bit 同步计数器（新版 DUT）
// 特性：同步复位、使能、回绕（255 -> 0）、进位输出（溢出时拉高一拍）
module counter8 #(
    parameter WIDTH = 8
)(
    input  wire             clk,
    input  wire             rst_n,   // 低有效同步复位
    input  wire             en,
    output reg  [WIDTH-1:0] count,
    output reg              carry    // 计数回绕时，下一拍为高
);

    always @(posedge clk) begin
        if (!rst_n) begin
            count <= {WIDTH{1'b0}};
            carry <= 1'b0;
        end else if (en) begin
            count <= count + 1'b1;
            carry <= (count == {WIDTH{1'b1}});   // 255 时提前置位，下一拍为高
        end else begin
            carry <= 1'b0;
        end
    end

endmodule