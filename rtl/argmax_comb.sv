import nn_pkg::*;
`timescale 1ns/1ps

module argmax_comb(
    input  acc_t logits [OUTPUT_SIZE],
    output logic [$clog2(OUTPUT_SIZE)-1:0] class_idx,
    output logic [OUTPUT_SIZE-1:0] onehot
);
    logic [$clog2(OUTPUT_SIZE)-1:0] max_idx;
    acc_t max_val;

    always_comb begin
        max_idx = '0;
        max_val = logits[0];

        for (int i = 1; i < OUTPUT_SIZE; i++) begin
            if (logits[i] > max_val) begin
                max_val = logits[i];
                max_idx = i[$clog2(OUTPUT_SIZE)-1:0];
            end
        end
    end

    assign class_idx = max_idx;
   assign onehot = (OUTPUT_SIZE'(1) << max_idx);
endmodule