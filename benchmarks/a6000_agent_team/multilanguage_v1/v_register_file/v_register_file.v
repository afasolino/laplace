module v_register_file #(parameter WIDTH=16, parameter DEPTH=4) (
 input wire clk,input wire rst_n,input wire write_i,input wire [1:0]write_addr_i,
 input wire [WIDTH-1:0]write_data_i,input wire [(WIDTH/8)-1:0]write_strb_i,
 input wire [1:0]read_addr_i,output wire [WIDTH-1:0]read_data_o
);
 reg [WIDTH-1:0] memory [0:DEPTH-1]; integer index;
 assign read_data_o = write_i && write_addr_i==read_addr_i ? write_data_i : memory[read_addr_i];
 always @(posedge clk or negedge rst_n) begin
  if(!rst_n) begin for(index=0;index<DEPTH;index=index+1) memory[index]<=0; end
  else if(write_i) begin
   if(write_strb_i[0]) memory[write_addr_i][7:0]<=write_data_i[7:0];
   if(write_strb_i[1]) memory[write_addr_i][15:8]<=write_data_i[15:8];
  end
 end
endmodule
