module sv_w1c_event_bank #(parameter int WIDTH=8)(
 input logic clk,input logic rst_n,input logic[WIDTH-1:0]event_i,
 input logic write_i,input logic[WIDTH-1:0]write_data_i,input logic[(WIDTH+7)/8-1:0]write_strb_i,
 input logic[WIDTH-1:0]enable_i,output logic[WIDTH-1:0]pending_o,output logic irq_o
);
 logic[WIDTH-1:0]clear_mask;integer byte_index;
 always_comb begin clear_mask='0;for(byte_index=0;byte_index<(WIDTH+7)/8;byte_index++)if(write_strb_i[byte_index])clear_mask[byte_index*8 +: 8]=write_data_i[byte_index*8 +: 8];end
 /* Intentional seeded defect: a simultaneous new event must win over W1C. */
 always_ff @(posedge clk or negedge rst_n)begin if(!rst_n)pending_o<='0;else pending_o<=(pending_o|event_i)&~(write_i?clear_mask:'0);end
 assign irq_o=|(pending_o&enable_i);
endmodule
