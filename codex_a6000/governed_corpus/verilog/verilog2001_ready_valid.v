/* Release multilanguage-corpus-v1; SPDX-License-Identifier: MIT */
module verilog2001_ready_valid #(
    parameter WIDTH = 8
) (
    input wire clk,
    input wire rst_n,
    input wire in_valid,
    output wire in_ready,
    input wire [WIDTH-1:0] in_data,
    output wire out_valid,
    input wire out_ready,
    output wire [WIDTH-1:0] out_data
);
    reg full_q;
    reg [WIDTH-1:0] data_q;

    assign in_ready = ~full_q | out_ready;
    assign out_valid = full_q;
    assign out_data = data_q;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            full_q <= 1'b0;
            data_q <= {WIDTH{1'b0}};
        end else if (in_ready) begin
            full_q <= in_valid;
            if (in_valid) begin
                data_q <= in_data;
            end
        end
    end
endmodule
