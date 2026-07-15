module tb_public;
 logic clk=0,rst_n=0,clear_i=0,increment_i=0;logic[3:0]count_o;sv_asserted_counter dut(.*);always #5 clk=~clk;
 initial begin repeat(2)@(posedge clk);rst_n=1;increment_i=1;repeat(3)@(posedge clk);#1;if(count_o!=3)$fatal(1,"FAIL");$display("PASS");$finish;end
endmodule
