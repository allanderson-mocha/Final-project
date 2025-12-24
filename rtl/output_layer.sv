import nn_arch_pkg::*;
`timescale 1ns/1ps

module output_layer (
    input  logic clk,
    input  logic rst_n,
    input  logic start,

    input  acc_t  h_in[HIDDEN_SIZE],
    input  data_t weight[HIDDEN_SIZE][OUTPUT_SIZE],
    input  acc_t  bias[OUTPUT_SIZE],

    output acc_t logits[OUTPUT_SIZE],
    output logic finished
);

    typedef enum logic [1:0] {
        IDLE,
        ACCUM,
        NEXT_PAIR,
        DONE
    } state_t;
    state_t state;

    int unsigned hid_idx; 
    int unsigned out_pair;   

    acc_t acc0, acc1;

    function automatic acc_t weight_ext(input data_t w);
        weight_ext = acc_t'($signed(w));
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= IDLE;
            finished <= 1'b0;
            hid_idx  <= 0;
            out_pair <= 0;

            acc0 <= '0;
            acc1 <= '0;

            for (int i = 0; i < OUTPUT_SIZE; i++)
                logits[i] <= '0;
        end
        else begin
            finished <= 1'b0;

            case (state)
                IDLE: begin
                    if (start) begin
                        hid_idx  <= 0;
                        out_pair <= 0;

                        acc0 <= bias[0];
                        acc1 <= bias[1];

                        state <= ACCUM;
                    end
                end

                ACCUM: begin
                    acc0 <= acc0 +
                            $signed(h_in[hid_idx]) *
                            weight_ext(weight[hid_idx][out_pair*2]);

                    acc1 <= acc1 +
                            $signed(h_in[hid_idx]) *
                            weight_ext(weight[hid_idx][out_pair*2 + 1]);

                    if (hid_idx == HIDDEN_SIZE-1)
                        state <= NEXT_PAIR;
                    else
                        hid_idx <= hid_idx + 1;
                end

                NEXT_PAIR: begin
                    logits[out_pair*2]     <= acc0;
                    logits[out_pair*2 + 1] <= acc1;

                    if (out_pair == (OUTPUT_SIZE/2 - 1)) begin
                        state <= DONE;
                    end
                    else begin
                        out_pair <= out_pair + 1;
                        hid_idx  <= 0;

                        acc0 <= bias[out_pair*2 + 2];
                        acc1 <= bias[out_pair*2 + 3];

                        state <= ACCUM;
                    end
                end

                DONE: begin
                    finished <= 1'b1;
                    state    <= IDLE;
                end

            endcase
        end
    end
endmodule
