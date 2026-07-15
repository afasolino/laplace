module tb_public;
    reg clk=0,rst_n=0,in_valid=0,out_ready=0;reg[7:0]in_data=0;
    wire in_ready,out_valid;wire[7:0]out_data;
    v_ready_valid_buffer dut(clk,rst_n,in_valid,in_ready,in_data,out_valid,out_ready,out_data);always #5 clk=~clk;
    initial begin repeat(2)@(posedge clk);rst_n=1;@(negedge clk);in_valid=1;in_data=8'h33;
      @(negedge clk);in_valid=0;if(!out_valid||out_data!==8'h33)begin $display("FAIL");$finish(1);end
      out_ready=1;@(negedge clk);if(out_valid)begin $display("FAIL");$finish(1);end $display("PASS");$finish;end
endmodule
