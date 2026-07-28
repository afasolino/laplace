module counter (
    input logic clock,
    input logic reset_n,
    output logic [3:0] count
);
    always_ff @(posedge clock) begin
        if (!reset_n) count <= '0;
        else count <= count + 1'b1;
    end
endmodule

