module tb_rv_buffer;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic in_valid;
    logic in_ready;
    logic [7:0] in_data;
    logic out_valid;
    logic out_ready;
    logic [7:0] out_data;

    rv_buffer #(.WIDTH(8)) dut (.*);
    always #5 clk = ~clk;

    initial begin
        in_valid = 0;
        in_data = '0;
        out_ready = 0;
        repeat (2) @(posedge clk);
        rst_n = 1;
        @(negedge clk);
        in_valid = 1;
        in_data = 8'h5a;
        @(negedge clk);
        in_valid = 0;
        if (!out_valid || out_data !== 8'h5a) $fatal(1, "missing buffered item");
        out_ready = 1;
        @(negedge clk);
        if (out_valid) $fatal(1, "item not consumed");
        $display("PASS: public rv_buffer smoke");
        $finish;
    end
endmodule
