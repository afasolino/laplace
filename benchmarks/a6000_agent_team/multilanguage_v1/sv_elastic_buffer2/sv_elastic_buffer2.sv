module sv_elastic_buffer2 #(parameter int WIDTH=8)(
 input logic clk,input logic rst_n,input logic in_valid,output logic in_ready,
 input logic[WIDTH-1:0]in_data,output logic out_valid,input logic out_ready,
 output logic[WIDTH-1:0]out_data
);
 logic[WIDTH-1:0]data_q[0:1];logic[1:0]count_q;
 assign in_ready=count_q<2;assign out_valid=count_q!=0;assign out_data=data_q[0];
 always_ff @(posedge clk or negedge rst_n) begin
  if(!rst_n) begin count_q<=0;data_q[0]<='0;data_q[1]<='0;end
  else case({in_valid&&in_ready,out_valid&&out_ready})
   2'b10:begin if(count_q==0)data_q[0]<=in_data;else data_q[1]<=in_data;count_q<=count_q+1'b1;end
   2'b01:begin data_q[0]<=data_q[1];count_q<=count_q-1'b1;end
   /* Intentional seeded defect: a full simultaneous transfer loses the replacement item. */
   2'b11:begin data_q[0]<=count_q==1?in_data:data_q[1];end
   default:count_q<=count_q;
  endcase
 end
endmodule
