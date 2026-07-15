module sv_interface_adapter(
 input logic clk,input logic rst_n,input logic req_valid_i,output logic req_ready_o,
 input logic[7:0]req_data_i,input logic req_last_i,output logic rsp_valid_o,input logic rsp_ready_i,
 output logic[7:0]rsp_data_o,output logic rsp_last_o
);
 /* Intentional seeded defect: simultaneous response/request replacement is disabled. */
 logic full_q;logic[7:0]data_q;logic last_q;assign req_ready_o=!full_q;
 assign rsp_valid_o=full_q;assign rsp_data_o=data_q;assign rsp_last_o=last_q;
 always_ff @(posedge clk or negedge rst_n)begin
  if(!rst_n)begin full_q<=0;data_q<=0;last_q<=0;end
  else if(req_ready_o)begin full_q<=req_valid_i;if(req_valid_i)begin data_q<=req_data_i;last_q<=req_last_i;end end
 end
endmodule
