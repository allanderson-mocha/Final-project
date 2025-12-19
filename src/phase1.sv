`timescale 1ns/1ps

module phase1 #(
    parameter IN_DIM = 64,
    parameter DATA_W = 8,
    parameter ACC_W  = 64,
    parameter HIDDEN_SIZE = 8,
    parameter OUTPUT_SIZE = 10,
    parameter WEIGHT_W = 8
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire [DATA_W*IN_DIM-1:0] bus_in,
    input  wire                     start,
    output wire [3:0]               class_idx,
    output wire [OUTPUT_SIZE-1:0]   one_out,
    output wire                     hidden_all_done,
    output wire                     output_finished,
    output wire [ACC_W*HIDDEN_SIZE-1:0] hidden_out_dbg_flat,
    output wire [ACC_W*OUTPUT_SIZE-1:0] logits_dbg_flat
);

    // ---------------- Hidden Layer ----------------
    wire [HIDDEN_SIZE-1:0] hidden_done;
    wire signed [ACC_W-1:0] hidden_out [0:HIDDEN_SIZE-1];
	
	
	integer q;

    // Weights and bias
    reg [DATA_W-1:0] weight_unpacked [0:HIDDEN_SIZE-1][0:IN_DIM-1];
    reg [IN_DIM*DATA_W-1:0] weight [0:HIDDEN_SIZE-1];
    reg signed [ACC_W-1:0] bias [0:HIDDEN_SIZE-1];
	
	
	wire signed [ACC_W-1:0] logits [0:OUTPUT_SIZE-1];




    integer x, y;
    initial begin
        $readmemh("W1_q.mem", weight_unpacked);
        for (x = 0; x < HIDDEN_SIZE; x = x + 1)
            for (y = 0; y < IN_DIM; y = y + 1)
                weight[x][y*DATA_W +: DATA_W] = weight_unpacked[x][y];
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

    // ---------------- Output Layer ----------------
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

    wire output_layer_start = hidden_all_done & ~hidden_all_done_d;
	


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

    // ---------------- Argmax ----------------
    argmax_comb #(
        .ACC_W(ACC_W),
        .OUTPUT_SIZE(OUTPUT_SIZE)
    ) u_argmax (
        .logits(logits),
        .class_idx(class_idx),
        .onehot(one_out)
    );


endmodule


// ================== Hidden Node ==================
module hidden_node #(
    parameter IN_DIM = 64,
    parameter DATA_W = 8,
    parameter ACC_W  = 64
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    start,
    input  wire [IN_DIM*DATA_W-1:0] in,
    input  wire [IN_DIM*DATA_W-1:0] weight,
    input  wire [ACC_W-1:0]        bias,
    output reg  [ACC_W-1:0]        out,
    output reg                     done
);

    typedef enum logic [1:0] {IDLE, MULT_SUM, DONE_STATE} state_t;
    state_t state;

    reg [$clog2(IN_DIM)-1:0] idx;
    reg signed [DATA_W-1:0]  in_val;
    reg signed [DATA_W-1:0]  w_val;
    reg signed [ACC_W-1:0]   acc;

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

// ================== Output Layer ==================
module output_layer #(
    parameter HIDDEN_LAYER_SIZE = 8,
    parameter OUTPUT_SIZE = 10,
    parameter ACCUMULATOR_WIDTH = 64,
    parameter WEIGHT_WIDTH = 8
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    input  wire signed [ACCUMULATOR_WIDTH-1:0] h_in [0:HIDDEN_LAYER_SIZE-1],
    input  wire signed [WEIGHT_WIDTH-1:0] weight_matrix [0:HIDDEN_LAYER_SIZE*OUTPUT_SIZE-1],
    input  wire signed [ACCUMULATOR_WIDTH-1:0] bias_vector [0:OUTPUT_SIZE-1],
    output wire signed [ACCUMULATOR_WIDTH-1:0] logits [0:OUTPUT_SIZE-1],
    output reg finished
);

    integer i, j;
    reg signed [ACCUMULATOR_WIDTH-1:0] acc;
    reg signed [ACCUMULATOR_WIDTH-1:0] logits_reg [0:OUTPUT_SIZE-1];
    assign logits = logits_reg; // wire signed driven by internal reg

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            finished <= 0;
            for (i = 0; i < OUTPUT_SIZE; i = i + 1)
                logits_reg[i] <= 0;
        end else if (start) begin
            for (i = 0; i < OUTPUT_SIZE; i = i + 1) begin
                acc = bias_vector[i];
                for (j = 0; j < HIDDEN_LAYER_SIZE; j = j + 1)
                    acc = acc + $signed(h_in[j]) * $signed({{ACCUMULATOR_WIDTH-WEIGHT_WIDTH{weight_matrix[j*OUTPUT_SIZE+i][WEIGHT_WIDTH-1]}}, weight_matrix[j*OUTPUT_SIZE+i]});
                logits_reg[i] <= acc;
            end
            finished <= 1;
        end else finished <= 0;
    end

endmodule




// ================== Argmax ==================
module argmax_comb #(
    parameter ACC_W = 64,
    parameter OUTPUT_SIZE = 10
)(
    input  wire signed [ACC_W-1:0] logits [0:OUTPUT_SIZE-1],
    output wire [3:0] class_idx,
    output wire [OUTPUT_SIZE-1:0] onehot
);
    integer i;
    reg [3:0] max_idx;
    reg signed [ACC_W-1:0] max_val;

    always_comb begin
        max_idx = 0;
        max_val = logits[0];
        for (i = 1; i < OUTPUT_SIZE; i = i + 1) begin
            if (logits[i] > max_val) begin
                max_val = logits[i];
                max_idx = i;
            end
        end
    end

    assign class_idx = max_idx;
    assign onehot = (1'b1 << max_idx);
endmodule





