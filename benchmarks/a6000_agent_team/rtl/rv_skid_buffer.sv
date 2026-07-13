module rv_skid_buffer #(
    parameter int unsigned WIDTH = 8
) (
    input  logic             clk,
    input  logic             rst_n,
    input  logic             in_valid,
    output logic             in_ready,
    input  logic [WIDTH-1:0] in_data,
    output logic             out_valid,
    input  logic             out_ready,
    output logic [WIDTH-1:0] out_data
);
    logic             full_q;
    logic [WIDTH-1:0] data_q;

    initial begin
        if (WIDTH == 0) $fatal(1, "WIDTH must be greater than zero");
    end

    assign in_ready  = !full_q || out_ready;
    assign out_valid = full_q;
    assign out_data  = data_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            full_q <= 1'b0;
            data_q <= '0;
        end else if (in_ready) begin
            if (in_valid) begin
                full_q <= 1'b1;
                data_q <= in_data;
            end else if (out_ready) begin
                full_q <= 1'b0;
            end
        end
    end

`ifdef FORMAL
    always_ff @(posedge clk) begin
        if (rst_n && $past(out_valid && !out_ready)) begin
            assert(out_valid);
            assert(out_data == $past(out_data));
        end
    end
`endif
endmodule
