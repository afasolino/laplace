module tb_public;
 logic clk=0,rst_n=0,write_i=0;logic[7:0]event_i=0,write_data_i=0,enable_i='1;logic[0:0]write_strb_i=1;logic[7:0]pending_o;logic irq_o;
 sv_w1c_event_bank dut(.*);always #5 clk=~clk;
 initial begin repeat(2)@(posedge clk);rst_n=1;@(negedge clk);event_i=8'h1;@(negedge clk);event_i=0;if(!irq_o)$fatal(1,"FAIL");
  write_i=1;write_data_i=1;@(negedge clk);write_i=0;if(pending_o!=0)$fatal(1,"FAIL");$display("PASS");$finish;end
endmodule
