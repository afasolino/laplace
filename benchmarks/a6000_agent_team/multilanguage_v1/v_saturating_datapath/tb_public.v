module tb_public;
 reg signed[7:0]a_i,b_i;wire signed[7:0]sum_o;
 v_saturating_datapath dut(a_i,b_i,sum_o);
 initial begin a_i=10;b_i=20;#1;if(sum_o!==30)begin $display("FAIL");$finish(1);end
  a_i=120;b_i=20;#1;if(sum_o!==127)begin $display("FAIL");$finish(1);end $display("PASS");$finish;end
endmodule
