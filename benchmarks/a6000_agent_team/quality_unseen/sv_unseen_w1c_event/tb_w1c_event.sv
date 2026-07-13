module tb_w1c_event;
    logic clk = 0;
    logic rst_n = 0;
    logic event_i;
    logic write_i;
    logic [31:0] write_data_i;
    logic [3:0] write_strb_i;
    logic pending_o;
    logic irq_o;

    w1c_event dut (.*);
    always #5 clk = ~clk;

    initial begin
        event_i = 0;
        write_i = 0;
        write_data_i = '0;
        write_strb_i = '0;
        repeat (2) @(posedge clk);
        rst_n = 1;
        @(negedge clk);
        write_i = 1;
        write_data_i = 32'h1;
        write_strb_i = 4'b0001;
        @(negedge clk);
        write_i = 0;
        event_i = 1;
        @(negedge clk);
        event_i = 0;
        if (!pending_o || !irq_o) $fatal(1, "enabled event did not interrupt");
        $display("PASS public w1c event");
        $finish;
    end
endmodule

