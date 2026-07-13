module w1c_event (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        event_i,
    input  logic        write_i,
    input  logic [31:0] write_data_i,
    input  logic [3:0]  write_strb_i,
    output logic        pending_o,
    output logic        irq_o
);
    logic enabled_q;
    logic pending_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            enabled_q <= 1'b0;
            pending_q <= 1'b0;
        end else begin
            if (event_i) pending_q <= 1'b1;
            if (write_i && write_strb_i[0]) begin
                enabled_q <= write_data_i[0];
                if (write_data_i[1]) pending_q <= 1'b0;
            end
        end
    end

    assign pending_o = pending_q;
    assign irq_o = enabled_q;
endmodule

