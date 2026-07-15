module v_parameterized_fifo #(parameter WIDTH = 8, parameter DEPTH = 4) (
    input wire clk, input wire rst_n, input wire push_i, input wire pop_i,
    input wire [WIDTH-1:0] data_i, output wire [WIDTH-1:0] data_o,
    output wire full_o, output wire empty_o
);
    reg [WIDTH-1:0] memory [0:DEPTH-1];
    reg [1:0] write_q, read_q; reg [2:0] count_q;
    assign data_o = memory[read_q];
    assign full_o = (count_q == DEPTH); assign empty_o = (count_q == 0);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin write_q <= 0; read_q <= 0; count_q <= 0; end
        else begin
            if (push_i && !full_o) begin memory[write_q] <= data_i; write_q <= write_q + 1'b1; end
            if (pop_i && !empty_o) read_q <= read_q + 1'b1;
            case ({push_i && !full_o, pop_i && !empty_o})
                2'b10: count_q <= count_q + 1'b1;
                2'b01: count_q <= count_q - 1'b1;
                default: count_q <= count_q;
            endcase
        end
    end
endmodule
