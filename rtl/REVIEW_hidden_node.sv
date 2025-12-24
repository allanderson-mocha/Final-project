`timescale 1ns/1ps

module hidden_node #(
    parameter IN_DIM = 4,
    parameter DATA_W = 8,
    parameter ACC_W  = 16
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,
    input  wire [IN_DIM*DATA_W-1:0] in,
    input  wire [IN_DIM*DATA_W-1:0] weight,
    input  wire signed [ACC_W-1:0]  bias,
    output reg  signed [ACC_W-1:0]  out,
    output reg                      done
);

    typedef enum logic [1:0] {IDLE, MULT_SUM, DONE_STATE} state_t;
    state_t state;

    reg [$clog2(IN_DIM)-1:0] idx;
    reg signed [DATA_W-1:0] in_val, w_val;
    reg signed [ACC_W-1:0] acc;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            idx <= 0;
            acc <= 0;
            out <= 0;
            done <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    acc <= bias;
                    idx <= 0;
                    if (start) state <= MULT_SUM;
                end
                MULT_SUM: begin
                    in_val = in[idx*DATA_W +: DATA_W];
                    w_val  = weight[idx*DATA_W +: DATA_W];
                    acc <= acc + $signed({{ACC_W-DATA_W{in_val[DATA_W-1]}}, in_val}) *
                                 $signed({{ACC_W-DATA_W{w_val[DATA_W-1]}}, w_val});
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
endmodule
