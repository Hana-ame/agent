// 4-bit 同步计数器（DUT）
// 特性：同步复位、使能、计数回绕（15 -> 0）
module counter #(
    parameter WIDTH = 4
)(
    input  wire             clk,
    input  wire             rst_n,   // 低有效异步复位（此处用同步复位，见代码）
    input  wire             en,
    output reg [WIDTH-1:0]  count
);

    always @(posedge clk) begin
        if (!rst_n)
            count <= {WIDTH{1'b0}};
        else if (en)
            count <= count + 1'b1;
        // en=0 时保持
    end

endmodule