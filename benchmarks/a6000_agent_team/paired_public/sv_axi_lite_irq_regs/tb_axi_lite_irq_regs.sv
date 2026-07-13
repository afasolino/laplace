module tb_axi_lite_irq_regs;
    logic clk = 0;
    logic rst_n = 0;
    logic [3:0] s_awaddr;
    logic s_awvalid;
    logic s_awready;
    logic [31:0] s_wdata;
    logic [3:0] s_wstrb;
    logic s_wvalid;
    logic s_wready;
    logic [1:0] s_bresp;
    logic s_bvalid;
    logic s_bready;
    logic [3:0] s_araddr;
    logic s_arvalid;
    logic s_arready;
    logic [31:0] s_rdata;
    logic [1:0] s_rresp;
    logic s_rvalid;
    logic s_rready;
    logic irq_input;
    logic irq;

    axi_lite_irq_regs dut (.*);
    always #5 clk = ~clk;

    initial begin
        s_awaddr = 0; s_awvalid = 0; s_wdata = 0; s_wstrb = 4'hf; s_wvalid = 0;
        s_bready = 1; s_araddr = 0; s_arvalid = 0; s_rready = 1; irq_input = 0;
        repeat (2) @(posedge clk);
        rst_n = 1;
        @(negedge clk);
        s_awvalid = 1; s_wvalid = 1; s_wdata = 1;
        @(negedge clk);
        s_awvalid = 0; s_wvalid = 0;
        repeat (2) @(posedge clk);
        if (s_bresp !== 2'b00) $fatal(1, "basic write failed");
        $display("PASS: public axi-lite smoke");
        $finish;
    end
endmodule
