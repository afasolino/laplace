module v_dual_channel_fsm(
 input wire clk,input wire rst_n,input wire a_valid_i,output wire a_ready_o,
 input wire b_valid_i,output wire b_ready_o,output reg done_o
);
 reg a_seen_q;
 v_channel channel_a(a_valid_i,a_ready_o,!a_seen_q);
 v_channel channel_b(b_valid_i,b_ready_o,a_seen_q);
 always @(posedge clk or negedge rst_n) begin
  if(!rst_n) begin a_seen_q<=0;done_o<=0;end
  else begin
   if(a_valid_i && a_ready_o) a_seen_q<=1;
   if(b_valid_i && b_ready_o) begin a_seen_q<=0;done_o<=1;end else done_o<=0;
  end
 end
endmodule
