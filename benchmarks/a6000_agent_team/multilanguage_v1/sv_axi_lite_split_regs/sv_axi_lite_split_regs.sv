module sv_axi_lite_split_regs(
 input logic clk,input logic rst_n,input logic[3:0]awaddr,input logic awvalid,output logic awready,
 input logic[31:0]wdata,input logic[3:0]wstrb,input logic wvalid,output logic wready,
 output logic[1:0]bresp,output logic bvalid,input logic bready,
 input logic[3:0]araddr,input logic arvalid,output logic arready,
 output logic[31:0]rdata,output logic[1:0]rresp,output logic rvalid,input logic rready
);
 logic aw_have_q,w_have_q;logic[3:0]awaddr_q;logic[31:0]wdata_q,register_q;logic[3:0]wstrb_q;integer i;
 assign awready=!aw_have_q&&!bvalid;assign wready=!w_have_q&&!bvalid;assign arready=!rvalid;
 always_ff @(posedge clk or negedge rst_n)begin
  if(!rst_n)begin aw_have_q<=0;w_have_q<=0;bvalid<=0;rvalid<=0;register_q<=0;bresp<=0;rresp<=0;rdata<=0;end
  else begin
   if(awready&&awvalid)begin aw_have_q<=1;awaddr_q<=awaddr;end
   if(wready&&wvalid)begin w_have_q<=1;wdata_q<=wdata;wstrb_q<=wstrb;end
   if(aw_have_q&&w_have_q&&!bvalid)begin
    bvalid<=1;bresp<=awaddr_q==0?2'b00:2'b10;
    if(awaddr_q==0)for(i=0;i<4;i++)if(wstrb_q[i])register_q[i*8 +:8]<=wdata_q[i*8 +:8];
    aw_have_q<=0;w_have_q<=0;
   end else if(bvalid&&bready)bvalid<=0;
   /* Intentional seeded defect: an unmapped response must stay stable while stalled. */
   if(bvalid&&!bready)bresp<=2'b00;
   if(arvalid&&arready)begin rvalid<=1;rresp<=araddr==0?2'b00:2'b10;rdata<=araddr==0?register_q:0;end
   else if(rvalid&&rready)rvalid<=0;
  end
 end
endmodule
