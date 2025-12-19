`timescale 1ns/1ps

module mlp_hidden_layer #(
    parameter IN_DIM = 4,
    parameter DATA_W = 8,
    parameter ACC_W  = 16,
    parameter HIDDEN_SIZE = 2
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire [DATA_W*IN_DIM-1:0] bus_in,
    input  wire                     start,
    output wire [ACC_W*HIDDEN_SIZE-1:0] hidden_out_dbg_flat,
    output wire                     hidden_all_done
);

    wire [HIDDEN_SIZE-1:0] hidden_done;
    wire signed [ACC_W-1:0] hidden_out [0:HIDDEN_SIZE-1];

    // ---------------- Dummy Weights and Biases ----------------
    reg [DATA_W*IN_DIM-1:0] weight [0:HIDDEN_SIZE-1];
    reg signed [ACC_W-1:0] bias [0:HIDDEN_SIZE-1];

    initial begin
        weight[0] = {8'd1,8'd2,8'd3,8'd4};
        weight[1] = {8'd5,8'd6,8'd7,8'd8};
        bias[0]   = 16'd10;
        bias[1]   = 16'd20;
    end

    genvar i;
    generate
        for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin : HIDDEN_LAYER
            hidden_node #(
                .IN_DIM(IN_DIM),
                .DATA_W(DATA_W),
                .ACC_W(ACC_W)
            ) u_hidden (
                .clk(clk),
                .rst_n(rst_n),
                .start(start),
                .in(bus_in),
                .weight(weight[i]),
                .bias(bias[i]),
                .out(hidden_out[i]),
                .done(hidden_done[i])
            );
        end
    endgenerate

    assign hidden_all_done = &hidden_done;

    generate
        for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin : HIDDEN_DBG
            assign hidden_out_dbg_flat[(i+1)*ACC_W-1 -: ACC_W] = hidden_out[i];
        end
    endgenerate

endmodule
