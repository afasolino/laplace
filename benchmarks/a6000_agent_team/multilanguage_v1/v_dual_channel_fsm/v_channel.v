module v_channel(input wire valid_i,output wire ready_o,input wire enable_i);
 assign ready_o=enable_i && valid_i;
endmodule
