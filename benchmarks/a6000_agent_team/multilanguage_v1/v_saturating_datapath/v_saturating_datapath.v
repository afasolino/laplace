module v_saturating_datapath #(parameter WIDTH=8) (
 input wire signed [WIDTH-1:0]a_i,input wire signed [WIDTH-1:0]b_i,
 output reg signed [WIDTH-1:0]sum_o
);
 reg signed [WIDTH:0] wide;
 always @* begin
  wide=a_i+b_i;
  if(wide > 127) sum_o=127;
  else if(wide < -128) sum_o=-128;
  else sum_o=wide[WIDTH-1:0];
 end
endmodule
