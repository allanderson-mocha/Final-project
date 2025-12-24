import nn_pkg::*;
`timescale 1ns/1ps

module network #(
    parameter IN_DIM = 64,
    parameter DATA_W = 8,
    parameter ACC_W  = 64,
    parameter HIDDEN_SIZE = 8,
    parameter OUTPUT_SIZE = 10,
    parameter WEIGHT_W = 8
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire [DATA_W*IN_DIM-1:0]     bus_in,
    input  wire                         start,
    output wire [3:0]                   class_idx,
    output wire [OUTPUT_SIZE-1:0]       one_out,
    output wire                         hidden_all_done,
    output wire                         output_finished,
    output wire [ACC_W*HIDDEN_SIZE-1:0] hidden_out_dbg_flat,
    output wire [ACC_W*OUTPUT_SIZE-1:0] logits_dbg_flat
);

    ////////////////
    //Hidden Layer//
    ////////////////
    logic hidden_done [HIDDEN_SIZE];
    logic signed [ACC_W-1:0] hidden_out [HIDDEN_SIZE];
	

    // Weights and bias
    logic signed [DATA_W-1:0] weight [HIDDEN_SIZE][IN_DIM];
    logic signed [ACC_W-1:0] bias [HIDDEN_SIZE];
	logic signed [ACC_W-1:0] logits [OUTPUT_SIZE];

    // Load mem file values
    integer x, y;
    initial begin
        $readmemh("W1_q.mem", weight);
        $readmemh("b1_q.mem", bias);
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

    // Flatten hidden outputs
    generate
        for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin : HIDDEN_DBG
            assign hidden_out_dbg_flat[(i+1)*ACC_W-1 -: ACC_W] = hidden_out[i];
        end
    endgenerate

    ////////////////
    //Output Layer//
    ////////////////
    reg signed [WEIGHT_W-1:0] weight_matrix [0:HIDDEN_SIZE*OUTPUT_SIZE-1];
    reg signed [ACC_W-1:0] bias_vector [0:OUTPUT_SIZE-1];
    

	integer k;
    initial begin
        $readmemh("W2_q.mem", weight_matrix);
        $readmemh("b2_q.mem", bias_vector);
		// Sign-extend 32-bit to ACC_W bits
		for (k = 0; k < OUTPUT_SIZE; k = k + 1)
			bias_vector[k] = {{(ACC_W-32){bias_vector[k][31]}}, bias_vector[k][31:0]};
    end

    reg hidden_all_done_d;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) hidden_all_done_d <= 1'b0;
        else hidden_all_done_d <= hidden_all_done;
    end

    wire output_layer_start = hidden_all_done & ~hidden_all_done_d; // TODO: ENSURE PULSE SIGNAL
	


    output_layer #(
        .HIDDEN_LAYER_SIZE(HIDDEN_SIZE),
        .OUTPUT_SIZE(OUTPUT_SIZE),
        .WEIGHT_WIDTH(WEIGHT_W),
        .ACCUMULATOR_WIDTH(ACC_W)
    ) u_fc (
        .clk(clk),
        .rst_n(rst_n),
        .start(output_layer_start),
        .h_in(hidden_out),
        .weight_matrix(weight_matrix),
        .bias_vector(bias_vector),
        .logits(logits),
        .finished(output_finished)
    );

    // Flatten logits
    generate
        for (i = 0; i < OUTPUT_SIZE; i = i + 1) begin : LOGITS_DBG
            assign logits_dbg_flat[(i+1)*ACC_W-1 -: ACC_W] = logits[i];
        end
    endgenerate

    // Argmax
    argmax_comb #(
        .ACC_W(ACC_W),
        .OUTPUT_SIZE(OUTPUT_SIZE)
    ) u_argmax (
        .logits(logits),
        .class_idx(class_idx),
        .onehot(one_out)
    );


endmodule

