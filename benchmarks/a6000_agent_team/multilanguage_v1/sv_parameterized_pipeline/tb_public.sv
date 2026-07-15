module tb_public;
 logic clk=0,rst_n=0,in_valid=0,out_ready=1;logic[7:0]in_data=0;logic in_ready,out_valid;logic[7:0]out_data;
 sv_parameterized_pipeline dut(.*);always #5 clk=~clk;
 initial begin repeat(2)@(posedge clk);rst_n=1;@(negedge clk);in_valid=1;in_data=8'h77;
  @(negedge clk);in_valid=0;if(!out_valid||out_data!==8'h77)$fatal(1,"FAIL");$display("PASS");$finish;end
endmodule
