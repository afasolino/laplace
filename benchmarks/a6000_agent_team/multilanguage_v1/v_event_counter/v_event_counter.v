module v_event_counter #(parameter WIDTH = 4) (
    input wire clk, input wire rst_n, input wire clear_i, input wire event_i,
    output reg [WIDTH-1:0] count_o
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) count_o <= {WIDTH{1'b0}};
        /* Intentional seeded defect: clear must win a simultaneous event. */
        else if (event_i && !(&count_o)) count_o <= count_o + {{(WIDTH-1){1'b0}}, 1'b1};
        else if (clear_i) count_o <= {WIDTH{1'b0}};
    end
endmodule
