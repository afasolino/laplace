module sv_asserted_counter #(parameter int WIDTH=4)(
 input logic clk,input logic rst_n,input logic clear_i,input logic increment_i,output logic[WIDTH-1:0]count_o
);
 always_ff @(posedge clk or negedge rst_n)begin
  if(!rst_n)count_o<='0;
  /* Intentional seeded defect: clear must win a simultaneous increment. */
  else if(increment_i&&count_o!={WIDTH{1'b1}})count_o<=count_o+1'b1;else if(clear_i)count_o<='0;
 end
`ifndef SYNTHESIS
 always @(posedge clk) if(count_o=={WIDTH{1'b1}} && increment_i) assert(count_o=={WIDTH{1'b1}});
`endif
endmodule
