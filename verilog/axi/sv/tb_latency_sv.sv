`timescale 1ns/1ps
module tb_latency_sv #(parameter int WAITC = 2);
  localparam int AW=8, DW=32;
  logic ACLK=0, ARESETn=0;
  always #5 ACLK=~ACLK;
  axi_lite_if #(.AW(AW),.DW(DW)) axi(ACLK);
  apb_if      #(.AW(AW),.DW(DW)) apb(ACLK);
  axi_apb_bridge_sv #(.AW(AW),.DW(DW)) bridge(
    .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb));
  apb_slave_regs_sv #(.AW(AW),.DW(DW),.NREG(16),.WAIT_CYCLES(WAITC)) slave(
    .PCLK(ACLK), .PRESETn(ARESETn), .apb(apb));
  axi_apb_monitor #(.AW(AW),.DW(DW),.NAME("LAT")) mon(
    .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb));

  int cyc = 0;
  always @(posedge ACLK) cyc <= cyc + 1;
  int errors = 0, checks = 0;

  // 写事务延迟测量（在 posedge 上采样，不加 #1）
  task automatic meas_wr(output int lat);
    int t0;
    @(posedge ACLK); t0 = cyc;
    axi.AWVALID <= 1'b1; axi.AWADDR <= 8'h00;
    axi.WVALID  <= 1'b1; axi.WDATA  <= 32'hA5A5;
    axi.BREADY  <= 1'b1;
    forever begin
      @(posedge ACLK);
      if (axi.AWREADY && axi.AWVALID) axi.AWVALID <= 1'b0;
      if (axi.WREADY  && axi.WVALID)  axi.WVALID  <= 1'b0;
      if (axi.BVALID) begin
        lat = cyc - t0;
        // 关键：BVALID 出现时桥尚未采样握手，BREADY 必须再保持一拍
        // （否则桥卡在 S_B_RESP —— 见 tb_handshake_protocol_sv）
        @(posedge ACLK);
        axi.BREADY <= 1'b0;
        @(posedge ACLK);
        disable meas_wr;
      end
    end
  endtask

  task automatic meas_rd(output int lat);
    int t0;
    @(posedge ACLK); t0 = cyc;
    axi.ARVALID <= 1'b1; axi.ARADDR <= 8'h00;
    axi.RREADY  <= 1'b1;
    forever begin
      @(posedge ACLK);
      if (axi.ARREADY && axi.ARVALID) axi.ARVALID <= 1'b0;
      if (axi.RVALID) begin
        lat = cyc - t0;
        @(posedge ACLK);
        axi.RREADY <= 1'b0;
        @(posedge ACLK);
        disable meas_rd;
      end
    end
  endtask

  initial begin
    $display("=== latency measurement (WAIT_CYCLES=%0d) ===", WAITC);
    ARESETn = 0;
    axi.AWVALID <= 0; axi.WVALID <= 0; axi.ARVALID <= 0;
    axi.BREADY <= 0; axi.RREADY <= 0;
    repeat(4) @(posedge ACLK);
    ARESETn = 1;
    @(posedge ACLK); cyc = 0;

    begin : measure
      int w_first, r_first, wl, rl;
      int w_max = 0, r_max = 0;
      bit jitter = 0;

      meas_wr(w_first);
      meas_rd(r_first);
      $display("  基线: 写 %0d 周期, 读 %0d 周期（WAIT_CYCLES=%0d）",
               w_first, r_first, WAITC);

      for (int i = 0; i < 10; i++) begin
        meas_wr(wl);
        meas_rd(rl);
        if (wl > w_max) w_max = wl;
        if (rl > r_max) r_max = rl;
        if (wl != w_first) jitter = 1;
        if (rl != r_first) jitter = 1;
      end

      // 检查 1：延迟恒定（无抖动）
      checks++;
      if (!jitter)
        $display("  ✓ 延迟恒定（10 轮无抖动）");
      else begin
        $display("  ✗ 延迟存在抖动：写 max=%0d base=%0d, 读 max=%0d base=%0d",
                 w_max, w_first, r_max, r_first);
        errors++;
      end

      // 检查 2：活性（延迟有上界，无死锁）
      //
      // ⚠ 上界不能写死 100：合法延迟 = 7 + WAIT_CYCLES，
      //   当 WAIT_CYCLES > 93 时就会超过 100，被误判为「疑似死锁」。
      //   实测 WAIT_CYCLES=255（从机 wait_cnt 是 8 位，最大值）时
      //   延迟 262 周期、公式正确，却被旧上界判成死锁。
      //   现改为「公式值 + 合理余量」，既能发现真死锁，
      //   又不会冤枉合法的慢速传输。
      checks++;
      if (w_max <= (7 + WAITC) + 5 && r_max <= (7 + WAITC) + 5)
        $display("  ✓ 无死锁（延迟 %0d 周期 <= 公式值 %0d + 余量 5）",
                 w_max, 7 + WAITC);
      else begin
        $display("  ✗ 延迟 %0d 周期，疑似死锁", w_max);
        errors++;
      end

      // 检查 3：延迟 = 7 + WAIT_CYCLES（结构可预测性）
      checks++;
      if (w_first == 7 + WAITC && r_first == 7 + WAITC)
        $display("  ✓ 延迟符合公式 7 + WAIT_CYCLES = %0d", 7 + WAITC);
      else begin
        $display("  ✗ 延迟 %0d 不符合公式 7+%0d=%0d",
                 w_first, WAITC, 7 + WAITC);
        errors++;
      end

      // 检查 4：读写延迟对称（串行桥对等处理）
      checks++;
      if (w_first == r_first)
        $display("  ✓ 读写延迟对称（均为 %0d 周期）", w_first);
      else begin
        $display("  ✗ 读写延迟不对称：写 %0d 读 %0d", w_first, r_first);
        errors++;
      end
    end

    $display("------------------------------------------------------");
    $display(" mon: %0d 次传输, %0d 违规", mon.xfer_count, mon.errors);
    $display(" latency: %0d checks, %0d errors", checks, errors);
    if (errors == 0) begin $display("PASS: latency baseline"); $finish; end
    else begin $display("FAIL"); $fatal(1); end
  end
endmodule
