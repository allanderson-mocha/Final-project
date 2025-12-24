`timescale 1ns/1ps

module mlp_output_layer #(
    parameter HIDDEN_SIZE = 8,
    parameter OUTPUT_SIZE = 10,
    parameter ACC_W = 64,
    parameter WEIGHT_W = 8
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire [ACC_W*HIDDEN_SIZE-1:0] hidden_in_flat,
    input  wire                     start,
    output wire [3:0]               class_idx,
    output wire [OUTPUT_SIZE-1:0]   one_out
);

    // Flattened weight/bias/logits
    reg signed [WEIGHT_W*HIDDEN_SIZE*OUTPUT_SIZE-1:0] weight_matrix_flat;
    reg signed [ACC_W*OUTPUT_SIZE-1:0] bias_vector_flat;
    wire signed [ACC_W*OUTPUT_SIZE-1:0] logits_flat;
    reg finished;

    integer k;

    // ---------------- Initialize weights/biases ----------------
    initial begin
        // Load weight/bias from memory files
        // Note: Use temporary arrays to flatten
        reg signed [WEIGHT_W-1:0] tmp_weights [0:HIDDEN_SIZE*OUTPUT_SIZE-1];
        reg signed [ACC_W-1:0] tmp_bias [0:OUTPUT_SIZE-1];

        $readmemh("W2_q.mem", tmp_weights);
        $readmemh("b2_q.mem", tmp_bias);

        // Flatten weights
        for (k = 0; k < HIDDEN_SIZE*OUTPUT_SIZE; k = k + 1) begin
            weight_matrix_flat[k*WEIGHT_W +: WEIGHT_W] = tmp_weights[k];
        end

        // Flatten bias
        for (k = 0; k < OUTPUT_SIZE; k = k + 1) begin
            // Sign-extend to ACC_W if needed
            bias_vector_flat[k*ACC_W +: ACC_W] = tmp_bias[k];
        end
    end

    // ---------------- Output Layer ----------------
    output_layer #(
        .HIDDEN_LAYER_SIZE(HIDDEN_SIZE),
        .OUTPUT_SIZE(OUTPUT_SIZE),
        .ACCUMULATOR_WIDTH(ACC_W),
        .WEIGHT_WIDTH(WEIGHT_W)
    ) u_fc (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .h_in_flat(hidden_in_flat),
        .weight_matrix_flat(weight_matrix_flat),
        .bias_vector_flat(bias_vector_flat),
        .logits_flat(logits_flat),
        .finished(finished)
    );

    // ---------------- Argmax (optional: ensure module exists) ----------------
    // argmax_comb #(
    //     .ACC_W(ACC_W),
    //     .OUTPUT_SIZE(OUTPUT_SIZE)
    // ) u_argmax (
    //     .logits_flat(logits_flat),
    //     .class_idx(class_idx),
    //     .onehot(one_out)
    // );

endmodule
