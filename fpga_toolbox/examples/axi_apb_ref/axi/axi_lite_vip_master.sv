// AXI4-Lite master 激励模型（test bench 用）
// 用 task 发起读/写事务，发送合法握手（VALID 需等 READY）。
// 支持等待模式：写事务可 AW/W 分开发或同发；读事务单发。
`timescale 1ns / 1ps

module axi_lite_vip_master #(
    parameter AW = 8,
    parameter DW = 32
)(
    input  wire          ACLK,
    input  wire          ARESETn,

    // 写地址
    output reg           AWVALID,
    output reg  [AW-1:0] AWADDR,
    input  wire          AWREADY,

    // 写数据
    output reg           WVALID,
    output reg  [DW-1:0] WDATA,
    input  wire          WREADY,

    // 写响应
    input  wire          BVALID,
    input  wire  [1:0]   BRESP,
    output reg           BREADY,

    // 读地址
    output reg           ARVALID,
    output reg  [AW-1:0] ARADDR,
    input  wire          ARREADY,

    // 读数据
    input  wire          RVALID,
    input  wire  [DW-1:0] RDATA,
    input  wire  [1:0]   RRESP,
    output reg           RREADY
);

    // ---- 初始化 ----
    initial begin
        AWVALID = 1'b0; AWADDR = {AW{1'b0}};
        WVALID  = 1'b0; WDATA  = {DW{1'b0}};
        BREADY  = 1'b0;
        ARVALID = 1'b0; ARADDR = {AW{1'b0}};
        RREADY  = 1'b1;   // 读数据总是就绪（简单 master）
    end

    // ---- 写事务：分开发（AW 先，W 后） ----
    task write_split(input [AW-1:0] addr, input [DW-1:0] data, output [1:0] resp);
        begin
            // AW 握手
            @(posedge ACLK);
            AWVALID <= 1'b1; AWADDR <= addr;
            do @(posedge ACLK); while (!AWREADY);
            AWVALID <= 1'b0;

            // W 握手
            @(posedge ACLK);
            WVALID <= 1'b1; WDATA <= data;
            do @(posedge ACLK); while (!WREADY);
            WVALID <= 1'b0;

            // B 握手
            BREADY <= 1'b1;
            do @(posedge ACLK); while (!BVALID);
            resp = BRESP;
            BREADY <= 1'b0;
            @(posedge ACLK);
        end
    endtask

    // ---- 写事务：AW/W 同发 ----
    task write_same(input [AW-1:0] addr, input [DW-1:0] data, output [1:0] resp);
        begin
            @(posedge ACLK);
            AWVALID <= 1'b1; AWADDR <= addr;
            WVALID  <= 1'b1; WDATA  <= data;
            // 等两个都握手成功
            while (!(AWREADY && WREADY)) @(posedge ACLK);
            AWVALID <= 1'b0; WVALID <= 1'b0;
            // 与 write_split 相同的 B 响应流程
            BREADY <= 1'b1;
            do @(posedge ACLK); while (!BVALID);
            resp = BRESP;
            BREADY <= 1'b0;
            @(posedge ACLK);
        end
    endtask

    // ---- 读事务 ----
    task read(input [AW-1:0] addr, output [DW-1:0] data, output [1:0] resp);
        begin
            @(posedge ACLK);
            ARVALID <= 1'b1; ARADDR <= addr;
            do @(posedge ACLK); while (!ARREADY);
            ARVALID <= 1'b0;

            RREADY <= 1'b1;
            do @(posedge ACLK); while (!RVALID);
            data = RDATA; resp = RRESP;
            RREADY <= 1'b0;
            @(posedge ACLK);
        end
    endtask

endmodule