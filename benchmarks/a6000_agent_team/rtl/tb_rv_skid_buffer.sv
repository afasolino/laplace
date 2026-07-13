module tb_rv_skid_buffer;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic in_valid;
    logic in_ready;
    logic [7:0] in_data;
    logic out_valid;
    logic out_ready;
    logic [7:0] out_data;

    rv_skid_buffer #(.WIDTH(8)) dut (
        .clk, .rst_n, .in_valid, .in_ready, .in_data, .out_valid, .out_ready, .out_data
    );

    always #5 clk = ~clk;

    task automatic check_condition(input logic condition, input string message);
        if (!condition) $fatal(1, "%s", message);
    endtask

    initial begin
        in_valid = 1'b0;
        in_data = '0;
        out_ready = 1'b0;
        repeat (2) @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);
        check_condition(!out_valid, "reset must empty the buffer");

        in_valid = 1'b1;
        in_data = 8'hA5;
        @(negedge clk);
        in_valid = 1'b0;
        check_condition(out_valid && out_data == 8'hA5, "accepted input must be retained");
        check_condition(!in_ready, "full buffer must apply backpressure");
        repeat (2) begin
            @(negedge clk);
            check_condition(out_valid && out_data == 8'hA5, "backpressure must keep valid and data stable");
        end

        out_ready = 1'b1;
        in_valid = 1'b1;
        in_data = 8'h3C;
        @(negedge clk);
        in_valid = 1'b0;
        check_condition(out_valid && out_data == 8'h3C, "simultaneous dequeue/enqueue must replace data");
        @(negedge clk);
        check_condition(!out_valid, "handshaken final item must clear the buffer");
        $display("PASS: rv_skid_buffer self-checking simulation");
        $finish;
    end
endmodule
