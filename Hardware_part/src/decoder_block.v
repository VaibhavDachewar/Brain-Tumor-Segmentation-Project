`timescale 1ns/1ps

module decoder_block #(
    parameter IN_HEIGHT = 16,        // Input height (from previous decoder or bottleneck)
    parameter IN_WIDTH = 16,         // Input width
    parameter IN_CHANNELS = 1024,    // Input channels
    parameter SKIP_CHANNELS = 512,   // Skip connection channels
    parameter OUT_CHANNELS = 512,    // Output channels
    parameter DATA_WIDTH = 16,
    parameter FRAC_WIDTH = 8
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    
    // Main input (from previous decoder or bottleneck)
    input wire [DATA_WIDTH-1:0] feature_in,
    input wire feature_valid,
    
    // Skip connection input
    input wire [DATA_WIDTH-1:0] skip_in,
    input wire skip_valid,
    
    // BatchNorm parameters
    input wire [DATA_WIDTH-1:0] gamma,
    input wire [DATA_WIDTH-1:0] beta,
    input wire [DATA_WIDTH-1:0] mean,
    input wire [DATA_WIDTH-1:0] variance,
    
    // Kernel weights for transpose convolution
    input wire [DATA_WIDTH-1:0] kernel_weight,
    input wire [DATA_WIDTH-1:0] bias,
    
    output wire [DATA_WIDTH-1:0] feature_out,
    output wire feature_valid_out,
    output wire decoder_done
);

// Calculate output dimensions after transpose convolution
localparam OUT_HEIGHT = IN_HEIGHT * 2;  // Stride = 2
localparam OUT_WIDTH = IN_WIDTH * 2;

// Internal signals
wire [DATA_WIDTH-1:0] transpose_out, concat_out, conv_out;
wire transpose_valid, concat_valid, conv_valid;
wire transpose_done, concat_done, conv_done;

// State machine
reg [2:0] state;
localparam IDLE = 3'b000;
localparam TRANSPOSE = 3'b001;
localparam CONCATENATE = 3'b010;
localparam CONV_BLOCK = 3'b011;
localparam DONE = 3'b100;

reg start_transpose, start_concat, start_conv;
reg decoder_done_reg;

// Conv2DTranspose Layer
conv2d_transpose #(
    .IN_HEIGHT(IN_HEIGHT),
    .IN_WIDTH(IN_WIDTH),
    .IN_CHANNELS(IN_CHANNELS),
    .OUT_CHANNELS(OUT_CHANNELS),
    .KERNEL_SIZE(2),
    .STRIDE(2),
    .DATA_WIDTH(DATA_WIDTH),
    .FRAC_WIDTH(FRAC_WIDTH)
) transpose_inst (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_transpose),
    .feature_in(feature_in),
    .feature_valid(feature_valid && (state == TRANSPOSE)),
    .kernel_weight(kernel_weight),
    .bias(bias),
    .feature_out(transpose_out),
    .feature_valid_out(transpose_valid),
    .transpose_done(transpose_done)
);

// Feature Concatenation
feature_concatenate #(
    .HEIGHT(OUT_HEIGHT),
    .WIDTH(OUT_WIDTH),
    .IN1_CHANNELS(OUT_CHANNELS),     // From transpose convolution
    .IN2_CHANNELS(SKIP_CHANNELS),    // From skip connection
    .DATA_WIDTH(DATA_WIDTH)
) concat_inst (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_concat),
    .feature_in1(transpose_out),
    .feature_valid1(transpose_valid && (state == CONCATENATE)),
    .feature_in2(skip_in),
    .feature_valid2(skip_valid && (state == CONCATENATE)),
    .feature_out(concat_out),
    .feature_valid_out(concat_valid),
    .concat_done(concat_done)
);

// Conv Block (Conv2D -> BN -> ReLU -> Conv2D -> BN -> ReLU)
conv_block #(
    .IMG_HEIGHT(OUT_HEIGHT),
    .IMG_WIDTH(OUT_WIDTH),
    .IN_CHANNELS(OUT_CHANNELS + SKIP_CHANNELS),
    .OUT_CHANNELS(OUT_CHANNELS),
    .DATA_WIDTH(DATA_WIDTH),
    .FRAC_WIDTH(FRAC_WIDTH)
) conv_block_inst (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_conv),
    .feature_in(concat_out),
    .feature_valid(concat_valid && (state == CONV_BLOCK)),
    .gamma(gamma),
    .beta(beta),
    .mean(mean),
    .variance(variance),
    .feature_out(conv_out),
    .feature_valid_out(conv_valid),
    .conv_block_done(conv_done)
);

// Output assignments
assign feature_out = conv_out;
assign feature_valid_out = conv_valid && (state == CONV_BLOCK);
assign decoder_done = decoder_done_reg;

// Control FSM
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        start_transpose <= 0;
        start_concat <= 0;
        start_conv <= 0;
        decoder_done_reg <= 0;
        $display("DEBUG: Decoder Block Reset");
        $display("Input: %0dx%0dx%0d -> Output: %0dx%0dx%0d", 
                 IN_HEIGHT, IN_WIDTH, IN_CHANNELS, OUT_HEIGHT, OUT_WIDTH, OUT_CHANNELS);
    end else begin
        case (state)
            IDLE: begin
                decoder_done_reg <= 0;
                if (start) begin
                    state <= TRANSPOSE;
                    start_transpose <= 1;
                    $display("DEBUG: Decoder Block started - TRANSPOSE phase");
                end
            end
            
            TRANSPOSE: begin
                start_transpose <= 0;
                if (transpose_done) begin
                    state <= CONCATENATE;
                    start_concat <= 1;
                    $display("DEBUG: Decoder Block TRANSPOSE->CONCATENATE transition");
                end
            end
            
            CONCATENATE: begin
                start_concat <= 0;
                if (concat_done) begin
                    state <= CONV_BLOCK;
                    start_conv <= 1;
                    $display("DEBUG: Decoder Block CONCATENATE->CONV_BLOCK transition");
                end
            end
            
            CONV_BLOCK: begin
                start_conv <= 0;
                if (conv_done) begin
                    state <= DONE;
                    decoder_done_reg <= 1;
                    $display("DEBUG: Decoder Block completed");
                end
            end
            
            DONE: begin
                decoder_done_reg <= 1;
            end
            
            default: begin
                state <= IDLE;
            end
        endcase
    end
end

// Debug: Monitor state changes
reg [2:0] prev_state = IDLE;
always @(posedge clk) begin
    if (state != prev_state) begin
        $display("DEBUG: Decoder Block State change to %s at time %0t", 
                 (state == IDLE) ? "IDLE" :
                 (state == TRANSPOSE) ? "TRANSPOSE" :
                 (state == CONCATENATE) ? "CONCATENATE" :
                 (state == CONV_BLOCK) ? "CONV_BLOCK" :
                 (state == DONE) ? "DONE" : "UNKNOWN", $time);
        prev_state <= state;
    end
end

endmodule
