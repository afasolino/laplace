module tb_public;
    reg clk = 0; reg rst_n = 0; reg clear_i = 0; reg event_i = 0;
    wire [3:0] count_o;
    v_event_counter dut(clk,rst_n,clear_i,event_i,count_o);
    always #5 clk = ~clk;
    initial begin
        repeat (2) @(posedge clk); rst_n = 1; event_i = 1;
        repeat (2) @(posedge clk); #1;
        if (count_o !== 4'd2) begin $display("FAIL"); $finish(1); end
        $display("PASS"); $finish;
    end
endmodule
