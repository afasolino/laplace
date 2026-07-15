module tb_public;
 reg clk=0,rst_n=0,write_i=0;reg[1:0]write_addr_i=0,read_addr_i=0;
 reg[15:0]write_data_i=0;reg[1:0]write_strb_i=0;wire[15:0]read_data_o;
 v_register_file dut(clk,rst_n,write_i,write_addr_i,write_data_i,write_strb_i,read_addr_i,read_data_o);
 always #5 clk=~clk;
 initial begin repeat(2)@(posedge clk);rst_n=1;@(negedge clk);write_i=1;write_data_i=16'h1234;write_strb_i=2'b11;
  @(negedge clk);write_i=0;if(read_data_o!==16'h1234)begin $display("FAIL");$finish(1);end $display("PASS");$finish;end
endmodule
