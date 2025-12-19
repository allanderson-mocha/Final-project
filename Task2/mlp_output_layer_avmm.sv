`timescale 1ns/1ps

module mlp_output_layer_avmm #(
    parameter HIDDEN_SIZE = 8,
    parameter OUTPUT_SIZE = 10,
    parameter ACC_W = 64,
    parameter WEIGHT_W = 8
)(
    input  wire                  clk,
    input  wire                  reset_n,

    // Avalon-MM Slave Interface
    input  wire [4:0]            avmm_address,
    input  wire [31:0]           avmm_writedata,
    input  wire                  avmm_write,
    input  wire                  avmm_read,
    output reg  [31:0]           avmm_readdata,
    output reg                   avmm_waitrequest,

    // Flattened hidden inputs
    input  wire [ACC_W*HIDDEN_SIZE-1:0] h_in_flat,

    // Outputs for Quartus pin mapping
    output wire [31:0] logits_flat_0,
    output wire [31:0] logits_flat_1,
    output wire [31:0] logits_flat_2,
    output wire [31:0] logits_flat_3,
    output wire [31:0] logits_flat_4,
    output wire [31:0] logits_flat_5,
    output wire [31:0] logits_flat_6,
    output wire [31:0] logits_flat_7,
    output wire [31:0] logits_flat_8,
    output wire [31:0] logits_flat_9,
    output wire [3:0] class_idx,          // predicted class
    output wire [OUTPUT_SIZE-1:0] one_hot, // one-hot
    output wire finished_flag
);

    // ---------------- Internal Signals ----------------
    reg start;
    wire signed [WEIGHT_W*HIDDEN_SIZE*OUTPUT_SIZE-1:0] weight_matrix_flat;
    wire signed [ACC_W*OUTPUT_SIZE-1:0] bias_vector_flat;
    wire [ACC_W*OUTPUT_SIZE-1:0] logits_flat_internal;

    // ---------------- Output Layer ----------------
    output_layer #(
        .HIDDEN_LAYER_SIZE(HIDDEN_SIZE),
        .OUTPUT_SIZE(OUTPUT_SIZE),
        .ACCUMULATOR_WIDTH(ACC_W),
        .WEIGHT_WIDTH(WEIGHT_W)
    ) u_fc (
        .clk(clk),
        .rst_n(reset_n),
        .start(start),
        .h_in_flat(h_in_flat),
        .weight_matrix_flat(weight_matrix_flat),
        .bias_vector_flat(bias_vector_flat),
        .logits_flat(logits_flat_internal),
        .finished(finished_flag)
    );

    // Split the wide logits into 32-bit chunks for Quartus
    assign logits_flat_0 = logits_flat_internal[31:0];
    assign logits_flat_1 = logits_flat_internal[63:32];
    assign logits_flat_2 = logits_flat_internal[95:64];
    assign logits_flat_3 = logits_flat_internal[127:96];
    assign logits_flat_4 = logits_flat_internal[159:128];
    assign logits_flat_5 = logits_flat_internal[191:160];
    assign logits_flat_6 = logits_flat_internal[223:192];
    assign logits_flat_7 = logits_flat_internal[255:224];
    assign logits_flat_8 = logits_flat_internal[287:256];
    assign logits_flat_9 = logits_flat_internal[319:288]; // adjust if ACC_W*OUTPUT_SIZE > 320

    // ---------------- Avalon-MM Register Mapping ----------------
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            start <= 0;
            avmm_readdata <= 0;
            avmm_waitrequest <= 0;
        end else begin
            avmm_readdata <= 0;
            avmm_waitrequest <= 0;

            // Write
            if (avmm_write) begin
                case (avmm_address)
                    5'h00: start <= avmm_writedata[0];
                    default: ;
                endcase
            end

            // Read
            if (avmm_read) begin
                case (avmm_address)
                    5'h00: avmm_readdata <= {31'b0, start};
                    default: avmm_readdata <= 32'h0;
                endcase
            end
        end
    end

    // ---------------- Argmax (Predicted Class & One-hot) ----------------
    argmax_comb #(
        .ACC_W(ACC_W),
        .OUTPUT_SIZE(OUTPUT_SIZE)
    ) u_argmax (
        .logits(logits_flat_internal[ACC_W*OUTPUT_SIZE-1 -: ACC_W*OUTPUT_SIZE]), 
        .class_idx(class_idx),
        .onehot(one_hot)
    );

endmodule
