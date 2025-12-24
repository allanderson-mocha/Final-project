/*
* NOTE:
* This module assumes a fixed MLP architecture.
* For changes in topology, modify 'nn_arch_pkg.sv'
*/

import nn_arch_pkg::IN_DIM;
import nn_arch_pkg::DATA_W;
import nn_arch_pkg::ACC_W;
import nn_arch_pkg::acc_t;
import nn_arch_pkg::data_t;

`timescale 1ns/1ps

module hidden_node (
    input  logic  clk,
    input  logic  rst_n,
    input  logic  start,
    input  data_t in[IN_DIM],  // Ensure input is unpacked when instantiating
    input  data_t weight[IN_DIM],
    input  acc_t  bias,

    output acc_t out,
    output logic done
);

    typedef enum logic [1:0] {
        IDLE, 
        MULT_SUM, 
        DONE_STATE
    } state_t;
    state_t state;

    localparam int IDX_W = $clog2(IN_DIM);
    logic [IDX_W-1:0] idx;
    data_t in_feat;
    data_t w_feat;
    acc_t acc;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            idx <= 0;
            acc <= 0;
            out <= 0;
            done <= 0;
        end else begin
            unique case (state)
                IDLE: begin
                    done <= 0;
                    acc <= bias;
                    idx <= 0;
                    if (start) state <= MULT_SUM;
                end

                MULT_SUM: begin
                    acc <= $signed({{ACC_W-DATA_W{in_feat[DATA_W-1]}}, in_feat}) * \
                           $signed({{ACC_W-DATA_W{w_feat[DATA_W-1]}}, w_feat}) + \
                           acc;
                    idx <= idx + 1;
                    if (idx == IN_DIM-1) state <= DONE_STATE;
                end

                DONE_STATE: begin
                    out <= (acc < 0) ? 0 : acc;
                    done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

    assign in_feat = in[idx*DATA_W +: DATA_W];
    assign w_feat  = weight[idx];

endmodule
