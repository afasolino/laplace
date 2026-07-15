module tb_public;
 reg clk=0,rst_n=0,req_valid_i=0,write_i=0,rsp_ready_i=1;reg[1:0]addr_i=0;reg[31:0]wdata_i=0;reg[3:0]wstrb_i=0;
 wire req_ready_o,rsp_valid_o;wire[31:0]rdata_o;wire error_o;
 v_register_bank_integration dut(clk,rst_n,req_valid_i,req_ready_o,write_i,addr_i,wdata_i,wstrb_i,rsp_valid_o,rsp_ready_i,rdata_o,error_o);
 always #5 clk=~clk;initial begin repeat(2)@(posedge clk);rst_n=1;@(negedge clk);req_valid_i=1;write_i=1;wdata_i=32'h55;wstrb_i=1;
  @(negedge clk);req_valid_i=0;write_i=0;@(negedge clk);req_valid_i=1;@(negedge clk);if(error_o)begin $display("FAIL");$finish(1);end
  $display("PASS");$finish;end
endmodule
