module rv_buffer #(
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
    logic             full;
    logic [WIDTH-1:0] data_q;

    initial begin
        if (WIDTH == 0) $error("WIDTH must be positive");
    end

    assign in_ready = !full;
    assign out_valid = full;
    assign out_data = data_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            full <= 1'b0;
        end else if (in_valid && in_ready) begin
            data_q <= in_data;
            full <= 1'b1;
        end else if (out_ready) begin
            full <= 1'b0;
        end
    end
endmodule
