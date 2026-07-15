module tb_public;
 reg clk=0,rst_n=0,a_valid_i=0,b_valid_i=0;wire a_ready_o,b_ready_o,done_o;
 v_dual_channel_fsm dut(clk,rst_n,a_valid_i,a_ready_o,b_valid_i,b_ready_o,done_o);always #5 clk=~clk;
 initial begin repeat(2)@(posedge clk);rst_n=1;@(negedge clk);a_valid_i=1;@(negedge clk);a_valid_i=0;b_valid_i=1;
  @(negedge clk);if(!done_o)begin $display("FAIL");$finish(1);end $display("PASS");$finish;end
endmodule
