module phase1 #(
    parameter IN_DIM = 64, parameter DATA_W = 8
)(
    input   wire                      clk,
    input   wire                      rst_n,    // Active-low reset
    input   wire [DATA_W*IN_DIM-1:0]  bus_in,
    input   wire                      start,
    output  wire [3:0]                class_idx,
    output  wire [9:0]                one_out
);

// Container size for MAC operations and bias
localparam ACC_W = 40;


// Internal signals
/*
PROJECT NOTE:
weights and bias can be converted to ports temporarily if we want to try out different values
in the testbench to find the best. If so, just comment them out below and redeclare them as 
wires instead of reg in the ports.
*/
wire [7:0] hidden_done;  // 8 node outputs
wire [7:0][DATA_W-1:0] hidden_out;   // 8 done flags (ideally only ever 0x00 or 0xFF)
reg [7:0][ACC_W-1:0] bias;  // 8 biases to be set
reg [7:0][IN_DIM*DATA_W-1:0] weight;    // 8 weight vectors to be set
// If we want to use the weights from the python files
//initial begin
//   $readmemh("W2_q.mem", weight_hidden);
//   $readmemh("b2_q.mem", bias_hidden);
// end


// Instantiation of 8 node hidden layer
genvar i;
generate;
    for (i = 0; i < 8; i++) begin : HIDDEN_LAYER
        hidden_node #(.IN_DIM(IN_DIM), .DATA_W(DATA_W), .ACC_W(ACC_W))
            node_i (
                .clk(clk),
                .rst_n(rst_n),
                .start(start),
                .in(bus_in),
                .bias(bias[i]),
                .weight(weight[i]),
                .out(hidden_out[i]),
                .done(hidden_done[i])
            );
    end
endgenerate

//flag to show hidden lower is finished, start output layer
wire hidden_all_done;
assign hidden_all_done = &hidden_done;
reg hidden_all_done_d;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            hidden_all_done_d <= 1'b0;
        else
            hidden_all_done_d <= hidden_all_done;
    end
wire output_layer_start;
assign output_layer_start = hidden_all_done & ~hidden_all_done_d; //rising edge signal

//instantiation of output layer
localparam HIDDEN_LAYER_SIZE = 8;
localparam INPUT_DATA_WIDTH = 8;
localparam OUTPUT_SIZE = 10;
localparam ACCUMULATOR_WIDTH = 32;
localparam WEIGHT_WIDTH = 8;
wire signed [ACCUMULATOR_WIDTH-1:0] logits [OUTPUT_SIZE];

reg signed [WEIGHT_WIDTH-1:0]       weight_matrix [0:HIDDEN_LAYER_SIZE*OUTPUT_SIZE-1];
reg signed [ACCUMULATOR_WIDTH-1:0]  bias_vector        [0:OUTPUT_SIZE-1];

 initial begin
   $readmemh("W2_q.mem", weight_matrix);
   $readmemh("b2_q.mem", bias_vector);
end

output_layer #(
        .HIDDEN_LAYER_SIZE(HIDDEN_LAYER_SIZE),
        .INPUT_DATA_WIDTH(INPUT_DATA_WIDTH),
        .OUTPUT_SIZE(OUTPUT_SIZE),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACCUMULATOR_WIDTH(ACCUMULATOR_WIDTH)
    ) u_fc (
        .clk    (clk),
        .rst  (rst_n),
        .start  (output_layer_start),
        .h_in   (hidden_out),
        .weight_matrix  (weight_matrix),
        .bias_vector     (bias_vector),
        .logits (logits)
    );
//instantiation of argmax
argmax #(
        .ACCUMULATOR_WIDTH (ACCUMULATOR_WIDTH),
        .OUTPUT_SIZE       (OUTPUT_SIZE)
    ) u_argmax (
        .logits       (logits),
        .class_idx    (class_idx),
        .class_onehot (one_out)
    );
    
endmodule

// Maybe create module for hidden layer node
module hidden_node #(
    parameter IN_DIM = 64, 
    parameter DATA_W = 8, 
    parameter ACC_W = 40
) (
    input   wire                      clk,
    input   wire                      rst_n,    // Active-low reset
    input   wire                      start,    // One cycle pulse
    input   wire [DATA_W*IN_DIM-1:0]  in,
    input   wire [ACC_W-1:0]          bias,
    input   wire [DATA_W*IN_DIM-1:0]  weight,
    
    output  reg [DATA_W-1:0]          out,  // In this case, 8 bit to be input for out layer
    output  reg                       done  // Flag for confirmation to start out layer
);

// FSM state definition
typedef enum logic [2:0] {
        IDLE = 3'd0,
        MULT = 3'd1,
        SUM = 3'd2,
        DONE = 3'd3
} state_t;

state_t state, next_state;

// Internal registers
reg [$clog2(IN_DIM)-1:0] idx;
reg [2*DATA_W-1:0]       product;
reg [ACC_W-1:0]          acc;

always @(posedge clk or negedge rst_n) begin
    // Active-Low reset
    if (!rst_n) begin
        state <= IDLE;
        acc <= 0;
        product <= 0;
        idx <= 0;
        done <= 1'b0;
    end else begin
        // FSM logic
        state <= next_state;

        case(state)
            IDLE: begin
                done <= 1'b0;
                // Wait for pulse
                if (start) begin
                    idx <= 0;
                    acc <= bias;
                end
            end
            
            MULT: begin
                product <= in[idx*DATA_W +: DATA_W] * weight[idx*DATA_W +: DATA_W];
            end

            SUM: begin
                acc <= acc + product;
                idx <= idx + 1;
            end

            DONE: begin
                done <= 1'b1;
                //ReLU
                if(acc[ACC_W-1] == 1'b1) begin
                    out <= 0;
                end else begin
                    out <= acc[DATA_W-1:0];
                end
            end

            default: ;
        endcase
    end
end

// Next state logic
always @(*) begin
    next_state = state;

    case (state)
        IDLE: begin
            if (start)
                next_state = MULT;
        end

        MULT: begin
            next_state = SUM;
        end

        SUM: begin
            if (idx == IN_DIM-1)
                next_state = DONE;
            else
                next_state = MULT;
        end

        DONE: begin
            if (!start)
                next_state = IDLE;
        end

        default: next_state = IDLE;
    endcase
end
endmodule

module output_layer #(
    parameter HIDDEN_LAYER_SIZE = 8,
    parameter INPUT_DATA_WIDTH = 16,
    parameter OUTPUT_SIZE = 10,
    parameter ACCUMULATOR_WIDTH = 32,
    parameter WEIGHT_WIDTH = 8
)
(
    input wire clk,
    input wire rst,
    input wire start,
    input wire signed [INPUT_DATA_WIDTH -1:0] h_in[HIDDEN_LAYER_SIZE],
    input wire signed [WEIGHT_WIDTH-1:0] weight_matrix [0: HIDDEN_LAYER_SIZE * OUTPUT_SIZE -1],
    input wire signed [ACCUMULATOR_WIDTH -1:0]bias_vector [0: OUTPUT_SIZE-1],
    output reg signed [ACCUMULATOR_WIDTH -1:0] logits[OUTPUT_SIZE]
);

// helps go grom flattened weight_vector to 2d W_idx
function automatic logic signed [WEIGHT_WIDTH-1:0] W_idx(
    input int i,  
    input int j   
);
    W_idx = weight_matrix[i*OUTPUT_SIZE + j];
endfunction

typedef enum logic [1:0] {
    IDLE,
    RUN,
    FINISH
} state_t;

state_t state, state_n;

reg [3:0] out_idx;
reg [3:0] in_idx;
reg signed [ACCUMULATOR_WIDTH-1:0] acc;

always @(posedge clk or negedge rst_n) begin
    if (!rst) begin
        state <= IDLE;
        out_idx <= '0;
        in_idx <= '0;
        acc <= '0;
    end
    else begin
        state <= state_n;

        case (state)
            IDLE: begin
                if (start) begin   
                    out_idx <= 0;
                    in_idx <= 0;
                    acc <= bias_vector[0];
                end
            end

            RUN: begin
                acc <= acc + $signed(h_in[in_idx]) * $signed(W_idx(in_idx, out_idx));
                if (in_idx == HIDDEN_LAYER_SIZE-1) begin
                    logits[out_idx] <= acc;
                    if (out_idx < OUTPUT_SIZE-1) begin
                        out_idx <= out_idx+1;
                        in_idx <= 0;
                        acc <= bias_vector[out_idx+1];
                    end
                end 
                else begin
                    in_idx <= in_idx + 1;
                end
            end

            FINISH: begin
            end
        endcase
    end
end
//next state logic
always @(*) begin
    state_n = state;

    case(state)
        IDLE: begin
            if (start) begin
                state_n = RUN;
            end
        end

        RUN: begin 
            if (in_idx == HIDDEN_LAYER_SIZE-1) begin
                if (out_idx == OUTPUT_SIZE-1)
                    state_n = FINISH;
            end
        end

        FINISH: begin  
            state_n = IDLE;
        end
    endcase
end
endmodule

module argmax #(
    parameter ACCUMULATOR_WIDTH = 32,
    parameter OUTPUT_SIZE = 10
)
(
    input wire signed [ACCUMULATOR_WIDTH-1: 0] logits [OUTPUT_SIZE],
    output reg [3:0] class_idx,
    output reg [OUTPUT_SIZE-1:0]  class_onehot
);

    integer i;
    reg signed [ACCUMULATOR_WIDTH-1:0] max_val;
    reg [3:0] max_idx;

   always @(*) begin
        class_onehot = '0;
        max_val = logits[0];
        max_idx = 4'd0;
        for (i = 1; i < 10; i++) begin
            if (logits[i] > max_val) begin
                max_val = logits[i];
                max_idx = i[3:0];
            end
        end
        class_idx = max_idx;
        class_onehot[max_idx] = 1'b1;
    end
endmodule
