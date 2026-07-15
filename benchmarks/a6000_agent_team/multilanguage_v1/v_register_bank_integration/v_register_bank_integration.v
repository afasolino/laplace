module v_register_bank_integration(
 input wire clk,input wire rst_n,input wire req_valid_i,output wire req_ready_o,
 input wire write_i,input wire [1:0]addr_i,input wire [31:0]wdata_i,input wire[3:0]wstrb_i,
 output reg rsp_valid_o,input wire rsp_ready_i,output reg[31:0]rdata_o,output reg error_o
);
 reg[31:0]register_q;assign req_ready_o=!rsp_valid_o || rsp_ready_i;
 always @(posedge clk or negedge rst_n) begin
  if(!rst_n) begin register_q<=0;rsp_valid_o<=0;rdata_o<=0;error_o<=0;end
  else begin
   if(rsp_valid_o && rsp_ready_i) rsp_valid_o<=0;
   if(req_valid_i && req_ready_o) begin
    rsp_valid_o<=1;error_o<=(addr_i!=0);rdata_o<=register_q;
    if(write_i && addr_i==0) begin
     if(wstrb_i[0])register_q[7:0]<=wdata_i[7:0];if(wstrb_i[1])register_q[15:8]<=wdata_i[15:8];
     if(wstrb_i[2])register_q[23:16]<=wdata_i[23:16];if(wstrb_i[3])register_q[31:24]<=wdata_i[31:24];
    end
   end
   /* Intentional seeded defect: an error response must remain stable while stalled. */
   if(!(req_valid_i && req_ready_o)) error_o<=0;
  end
 end
endmodule
