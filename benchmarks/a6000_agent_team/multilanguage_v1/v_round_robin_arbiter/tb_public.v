module tb_public;
 reg clk=0,rst_n=0,accept_i=0;reg[1:0]request_i=0;wire[1:0]grant_o;
 v_round_robin_arbiter dut(clk,rst_n,request_i,accept_i,grant_o);always #5 clk=~clk;
 initial begin repeat(2)@(posedge clk);rst_n=1;@(negedge clk);request_i=2'b11;
  if(grant_o!==2'b01)begin $display("FAIL");$finish(1);end accept_i=1;@(negedge clk);
  if(grant_o!==2'b10)begin $display("FAIL");$finish(1);end $display("PASS");$finish;end
endmodule
