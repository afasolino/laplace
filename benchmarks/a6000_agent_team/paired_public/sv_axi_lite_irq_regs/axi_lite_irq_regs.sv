module axi_lite_irq_regs #(
    parameter int unsigned ADDR_WIDTH = 4
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic [ADDR_WIDTH-1:0] s_awaddr,
    input  logic                  s_awvalid,
    output logic                  s_awready,
    input  logic [31:0]           s_wdata,
    input  logic [3:0]            s_wstrb,
    input  logic                  s_wvalid,
    output logic                  s_wready,
    output logic [1:0]            s_bresp,
    output logic                  s_bvalid,
    input  logic                  s_bready,
    input  logic [ADDR_WIDTH-1:0] s_araddr,
    input  logic                  s_arvalid,
    output logic                  s_arready,
    output logic [31:0]           s_rdata,
    output logic [1:0]            s_rresp,
    output logic                  s_rvalid,
    input  logic                  s_rready,
    input  logic                  irq_input,
    output logic                  irq
);
    localparam logic [1:0] RESP_OKAY = 2'b00;
    localparam logic [1:0] RESP_SLVERR = 2'b10;
    logic [31:0] control_q;
    logic status_q;

    assign s_awready = !s_bvalid;
    assign s_wready = !s_bvalid;
    assign s_arready = !s_rvalid;
    assign irq = status_q & control_q[0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            control_q <= '0;
            status_q <= 1'b0;
            s_bvalid <= 1'b0;
            s_bresp <= RESP_OKAY;
            s_rvalid <= 1'b0;
            s_rresp <= RESP_OKAY;
            s_rdata <= '0;
        end else begin
            if (irq_input) status_q <= 1'b1;
            if (s_awvalid && s_wvalid && s_awready && s_wready) begin
                s_bvalid <= 1'b1;
                if (s_awaddr == '0) begin
                    control_q <= s_wdata; // Seeded defect: WSTRB and independent channels ignored.
                    s_bresp <= RESP_OKAY;
                end else if (s_awaddr == 4) begin
                    status_q <= status_q & ~s_wdata[0];
                    s_bresp <= RESP_OKAY;
                end else begin
                    s_bresp <= RESP_SLVERR;
                end
            end
            if (s_bvalid && s_bready) s_bvalid <= 1'b0;
            if (s_arvalid && s_arready) begin
                s_rvalid <= 1'b1;
                if (s_araddr == '0) begin
                    s_rdata <= control_q;
                    s_rresp <= RESP_OKAY;
                end else if (s_araddr == 4) begin
                    s_rdata <= {31'b0, status_q};
                    s_rresp <= RESP_OKAY;
                end else begin
                    s_rdata <= '0;
                    s_rresp <= RESP_SLVERR;
                end
            end
            if (s_rvalid && s_rready) s_rvalid <= 1'b0;
        end
    end
endmodule
