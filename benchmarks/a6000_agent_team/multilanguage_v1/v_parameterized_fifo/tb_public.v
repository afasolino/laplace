module tb_public;
    reg clk=0,rst_n=0,push_i=0,pop_i=0; reg [7:0] data_i=0;
    wire [7:0] data_o; wire full_o,empty_o;
    v_parameterized_fifo dut(clk,rst_n,push_i,pop_i,data_i,data_o,full_o,empty_o); always #5 clk=~clk;
    initial begin
        repeat(2) @(posedge clk); rst_n=1; @(negedge clk); push_i=1; data_i=8'h5a;
        @(negedge clk); push_i=0; if(empty_o || data_o!==8'h5a) begin $display("FAIL");$finish(1);end
        pop_i=1; @(negedge clk); pop_i=0; if(!empty_o) begin $display("FAIL");$finish(1);end
        $display("PASS");$finish;
    end
endmodule
