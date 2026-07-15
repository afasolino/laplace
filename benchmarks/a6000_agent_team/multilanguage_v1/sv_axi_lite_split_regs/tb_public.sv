module tb_public;
 logic clk=0,rst_n=0;logic[3:0]awaddr=0,araddr=0;logic awvalid=0,wvalid=0,bready=1,arvalid=0,rready=1;
 logic[31:0]wdata=0;logic[3:0]wstrb=0;logic awready,wready,bvalid,arready,rvalid;logic[1:0]bresp,rresp;logic[31:0]rdata;
 sv_axi_lite_split_regs dut(.*);always #5 clk=~clk;
 initial begin repeat(2)@(posedge clk);rst_n=1;@(negedge clk);awvalid=1;wvalid=1;wdata=32'h12;wstrb=1;
  @(negedge clk);awvalid=0;wvalid=0;wait(bvalid);if(bresp!=0)$fatal(1,"FAIL");$display("PASS");$finish;end
endmodule
