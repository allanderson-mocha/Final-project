`timescale 1ns/1ps

module mlp_hidden_layer_avmm #(
    parameter IN_DIM = 4,
    parameter DATA_W = 8,
    parameter ACC_W  = 16,
    parameter HIDDEN_SIZE = 2
)(
    input  wire         clk,
    input  wire         reset_n,

    // Avalon-MM Slave Interface
    input  wire [3:0]   avmm_address,   // byte offset
    input  wire [31:0]  avmm_writedata,
    input  wire         avmm_write,
    input  wire         avmm_read,
    output reg  [31:0]  avmm_readdata,
    output reg          avmm_waitrequest
);

    // =========================
    // INTERNAL SIGNALS
    // =========================
    reg [DATA_W*IN_DIM-1:0] bus_in;
    reg start;
    wire [ACC_W*HIDDEN_SIZE-1:0] hidden_out_flat;
    wire hidden_done;

    // DEBUG COUNTER
    reg [31:0] debug_counter;

    // convert byte-address to word index
    wire [1:0] addr_word = avmm_address[3:2];

    // =========================
    // INSTANTIATE HIDDEN LAYER
    // =========================
    mlp_hidden_layer #(
        .IN_DIM(IN_DIM),
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .HIDDEN_SIZE(HIDDEN_SIZE)
    ) u_hidden_layer (
        .clk(clk),
        .rst_n(reset_n),
        .bus_in(bus_in),
        .start(start),
        .hidden_out_dbg_flat(hidden_out_flat),
        .hidden_all_done(hidden_done)
    );

    // =========================
    // AVMM READ/WRITE LOGIC
    // =========================
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            start            <= 1'b0;
            bus_in           <= '0;
            avmm_readdata    <= 32'h0;
            avmm_waitrequest <= 1'b0;
            debug_counter    <= 32'h0;
        end 
        else begin
            // increment debug counter every clock
            debug_counter <= debug_counter + 1;

            avmm_readdata    <= 32'h0;
            avmm_waitrequest <= 1'b0;  // always ready

            // default: clear start pulse each cycle
            start <= 1'b0;

            // =====================
            //      WRITE LOGIC
            // =====================
            if (avmm_write) begin
                case (addr_word)
                    2'd0: begin
                        // offset 0 -> start control
                        if (avmm_writedata[0])
                            start <= 1'b1;
                    end

                    default: begin
                        // word index 1..N -> bus_in chunks
                        integer wi;
                        wi = addr_word - 1;
                        bus_in[wi*32 +: 32] <= avmm_writedata;
                    end
                endcase
            end

            // =====================
            //      READ LOGIC
            // =====================
            if (avmm_read) begin
                case (addr_word)
                    2'd0: avmm_readdata <= debug_counter;           // debug counter
                    2'd1: avmm_readdata <= hidden_out_flat[31:0];  // hidden_out low word
                    2'd2: avmm_readdata <= {31'b0, hidden_done};   // done flag
                    default: avmm_readdata <= 32'hDEADCAFE;        // debug
                endcase
            end
        end
    end

endmodule
