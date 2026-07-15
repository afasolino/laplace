module tb_public;
 logic clk=0,rst_n=0,req_valid_i=0,req_last_i=0,rsp_ready_i=0;logic[7:0]req_data_i=0;
 logic req_ready_o,rsp_valid_o,rsp_last_o;logic[7:0]rsp_data_o;
 sv_interface_adapter dut(.*);always #5 clk=~clk;
 initial begin repeat(2)@(posedge clk);rst_n=1;@(negedge clk);req_valid_i=1;req_data_i=8'hca;req_last_i=1;
  @(negedge clk);req_valid_i=0;if(!rsp_valid_o||rsp_data_o!==8'hca||!rsp_last_o)$fatal(1,"FAIL");$display("PASS");$finish;end
endmodule
