module v_round_robin_arbiter (
    input wire clk,input wire rst_n,input wire [1:0] request_i,input wire accept_i,
    output reg [1:0] grant_o
);
    reg priority_q;
    always @* begin
        grant_o=2'b00;
        if(priority_q==1'b0) begin
            if(request_i[0]) grant_o=2'b01; else if(request_i[1]) grant_o=2'b10;
        end else begin
            if(request_i[1]) grant_o=2'b10; else if(request_i[0]) grant_o=2'b01;
        end
    end
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) priority_q<=0;
        /* Intentional seeded defect: accept without a grant must not rotate. */
        else if(accept_i) priority_q<=~priority_q;
    end
endmodule
