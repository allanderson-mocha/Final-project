`timescale 1ns/1ps

module tb_phase1;

    // Parameters
    parameter IN_DIM = 64;
    parameter DATA_W = 8;
    parameter ACC_W  = 64;
    parameter CLK_PERIOD = 10;

    // DUT signals
    reg clk;
    reg rst_n;
    reg start;
    reg [DATA_W*IN_DIM-1:0] bus_in;
    wire [3:0] class_idx;
    wire [9:0] one_out;
    wire hidden_all_done;
    wire output_finished;
    wire [ACC_W*8-1:0] hidden_out_dbg_flat;
    wire [ACC_W*10-1:0] logits_dbg_flat;

    // Unpacked arrays for easier access
    wire signed [ACC_W-1:0] hidden_out_dbg [0:7];
    wire signed [ACC_W-1:0] logits_dbg [0:9];

    // Clock generation
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Instantiate DUT
    phase1 #(
        .IN_DIM(IN_DIM),
        .DATA_W(DATA_W),
        .ACC_W(ACC_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .bus_in(bus_in),
        .class_idx(class_idx),
        .one_out(one_out),
        .hidden_all_done(hidden_all_done),
        .output_finished(output_finished),
        .hidden_out_dbg_flat(hidden_out_dbg_flat),
        .logits_dbg_flat(logits_dbg_flat)
    );

    // Unpack flat wires
    genvar h, o;
    generate
        for (h = 0; h < 8; h = h + 1)
            assign hidden_out_dbg[h] = hidden_out_dbg_flat[(h+1)*ACC_W-1 -: ACC_W];
        for (o = 0; o < 10; o = o + 1)
            assign logits_dbg[o] = logits_dbg_flat[(o+1)*ACC_W-1 -: ACC_W];
    endgenerate

    // Test vectors
    reg [DATA_W*IN_DIM-1:0] test_vectors [0:3];

    // Expected values
    reg signed [ACC_W-1:0] expected_hidden [0:7];
    reg signed [ACC_W-1:0] expected_logits [0:9];

    // Task to compute expected outputs
    task compute_expected;
        input [DATA_W*IN_DIM-1:0] in_vector;
        integer i, j;
        reg signed [ACC_W-1:0] acc;
        reg signed [DATA_W-1:0] w, x;
        begin
            // Hidden layer
            for (i = 0; i < 8; i = i + 1) begin
                acc = dut.bias[i];
                for (j = 0; j < IN_DIM; j = j + 1) begin
                    w = dut.weight[i][j*DATA_W +: DATA_W];
                    x = in_vector[j*DATA_W +: DATA_W];
                    acc = acc + $signed({{ACC_W-DATA_W{w[DATA_W-1]}}, w}) *
                                $signed({{ACC_W-DATA_W{x[DATA_W-1]}}, x});
                end
                expected_hidden[i] = (acc < 0) ? 0 : acc; // ReLU
            end

            // Output layer
            for (i = 0; i < 10; i = i + 1) begin
                acc = dut.bias_vector[i];
                for (j = 0; j < 8; j = j + 1) begin
                    acc = acc + $signed({{ACC_W-DATA_W{dut.weight_matrix[j*10 + i][DATA_W-1]}}, dut.weight_matrix[j*10 + i]}) *
                                  $signed(expected_hidden[j]);
                end
                expected_logits[i] = acc;
            end
        end
    endtask

    // Initialize test vectors
    initial begin
        test_vectors[0] = {64{8'd0}}; // all zeros
		test_vectors[1] = {64{8'd1}}; // all ones
		test_vectors[2] = {
			8'd1,8'd2,8'd3,8'd4,8'd5,8'd6,8'd7,8'd8,
			8'd1,8'd2,8'd3,8'd4,8'd5,8'd6,8'd7,8'd8,
			8'd1,8'd2,8'd3,8'd4,8'd5,8'd6,8'd7,8'd8,
			8'd1,8'd2,8'd3,8'd4,8'd5,8'd6,8'd7,8'd8,
			8'd1,8'd2,8'd3,8'd4,8'd5,8'd6,8'd7,8'd8,
			8'd1,8'd2,8'd3,8'd4,8'd5,8'd6,8'd7,8'd8,
			8'd1,8'd2,8'd3,8'd4,8'd5,8'd6,8'd7,8'd8,
			8'd1,8'd2,8'd3,8'd4,8'd5,8'd6,8'd7,8'd8
		};
		test_vectors[3] = {
			8'd8,8'd7,8'd6,8'd5,8'd4,8'd3,8'd2,8'd1,
			8'd8,8'd7,8'd6,8'd5,8'd4,8'd3,8'd2,8'd1,
			8'd8,8'd7,8'd6,8'd5,8'd4,8'd3,8'd2,8'd1,
			8'd8,8'd7,8'd6,8'd5,8'd4,8'd3,8'd2,8'd1,
			8'd8,8'd7,8'd6,8'd5,8'd4,8'd3,8'd2,8'd1,
			8'd8,8'd7,8'd6,8'd5,8'd4,8'd3,8'd2,8'd1,
			8'd8,8'd7,8'd6,8'd5,8'd4,8'd3,8'd2,8'd1,
			8'd8,8'd7,8'd6,8'd5,8'd4,8'd3,8'd2,8'd1
		};
		test_vectors[4] = {
			8'd0,8'd1,8'd0,8'd1,8'd0,8'd1,8'd0,8'd1,
			8'd0,8'd1,8'd0,8'd1,8'd0,8'd1,8'd0,8'd1,
			8'd0,8'd1,8'd0,8'd1,8'd0,8'd1,8'd0,8'd1,
			8'd0,8'd1,8'd0,8'd1,8'd0,8'd1,8'd0,8'd1,
			8'd0,8'd1,8'd0,8'd1,8'd0,8'd1,8'd0,8'd1,
			8'd0,8'd1,8'd0,8'd1,8'd0,8'd1,8'd0,8'd1,
			8'd0,8'd1,8'd0,8'd1,8'd0,8'd1,8'd0,8'd1,
			8'd0,8'd1,8'd0,8'd1,8'd0,8'd1,8'd0,8'd1
		};

        // Reset
        rst_n = 0; start = 0; bus_in = 0;
        #(CLK_PERIOD*2);
        rst_n = 1;

        // Run tests
        for (integer t = 0; t < 4; t = t + 1) begin
            bus_in = test_vectors[t];
            @(posedge clk);
            start = 1;
            @(posedge clk);
            start = 0;

            // Wait for output to finish
            wait(output_finished);
			@(posedge clk);

            // Compute expected outputs only for first test
            if (t == 0) compute_expected(test_vectors[t]);

            // Display
            $display("\n=== Test %0d ===", t+1);
            $write("Hidden outputs (DUT): ");
            for (integer j = 0; j < 8; j = j + 1) $write("%0d ", hidden_out_dbg[j]);
            $write("\n");
            if (t == 0) begin
                $write("Hidden expected     : ");
                for (integer j = 0; j < 8; j = j + 1) $write("%0d ", expected_hidden[j]);
                $write("\n");
            end

            $write("Logits (DUT)        : ");
            for (integer j = 0; j < 10; j = j + 1) $write("%0d ", logits_dbg[j]);
            $write("\n");
            if (t == 0) begin
                $write("Logits expected     : ");
                for (integer j = 0; j < 10; j = j + 1) $write("%0d ", expected_logits[j]);
                $write("\n");
            end

            $display("Predicted class = %0d", class_idx);
            $display("One-hot output  = %b", one_out);

            #(CLK_PERIOD*5);
        end

        $display("\nAll tests finished.");
    end

    // Dump waves
    initial begin
        $dumpfile("tb_phase1.vcd");
        $dumpvars(0, tb_phase1);
    end

endmodule
