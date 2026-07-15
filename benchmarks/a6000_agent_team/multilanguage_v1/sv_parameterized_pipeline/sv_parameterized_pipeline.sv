module sv_parameterized_pipeline #(parameter int WIDTH=8, parameter int STAGES=1)(
 input logic clk,input logic rst_n,input logic in_valid,output logic in_ready,input logic[WIDTH-1:0]in_data,
 output logic out_valid,input logic out_ready,output logic[WIDTH-1:0]out_data
);
 /* Intentional seeded defect: STAGES is ignored and only one stage is implemented. */
 logic valid_q;logic[WIDTH-1:0]data_q;assign in_ready=!valid_q||out_ready;assign out_valid=valid_q;assign out_data=data_q;
 always_ff @(posedge clk or negedge rst_n)begin if(!rst_n)begin valid_q<=0;data_q<='0;end else if(in_ready)begin valid_q<=in_valid;if(in_valid)data_q<=in_data;end end
endmodule
