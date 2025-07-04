`timescale 1ns/1ps

module encoder_block #(
    parameter IMG_HEIGHT = 256,
    parameter IMG_WIDTH = 256,
    parameter IN_CHANNELS = 3,          // <-- NEW: For first encoder block
    parameter OUT_CHANNELS = 64,        // <-- NEW: For output after conv
    parameter DATA_WIDTH = 16,
    parameter FRAC_WIDTH = 8
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [DATA_WIDTH-1:0] feature_in,
    input wire feature_valid,
    
    // BatchNorm parameters
    input wire [DATA_WIDTH-1:0] gamma,
    input wire [DATA_WIDTH-1:0] beta,
    input wire [DATA_WIDTH-1:0] mean,
    input wire [DATA_WIDTH-1:0] variance,
    
    // Skip connection output (before pooling)
    output wire [DATA_WIDTH-1:0] skip_out,
    output wire skip_valid,
    
    // Pooled output
    output wire [DATA_WIDTH-1:0] pooled_out,
    output wire pooled_valid,
    
    output wire encoder_done
);

// Internal signals
wire [DATA_WIDTH-1:0] conv_out;
wire conv_valid, conv_done;
wire pool_done;

// State machine
reg [1:0] state;
localparam IDLE = 2'b00;
localparam CONV = 2'b01;
localparam POOL = 2'b10;
localparam DONE = 2'b11;

reg start_conv, start_pool;
reg encoder_done_reg;

// Conv Block (Conv2D -> BN -> ReLU -> Conv2D -> BN -> ReLU)
conv_block #(
    .IMG_HEIGHT(IMG_HEIGHT),
    .IMG_WIDTH(IMG_WIDTH),
    .IN_CHANNELS(IN_CHANNELS),         // ✅ Pass input channels
    .OUT_CHANNELS(OUT_CHANNELS),       // ✅ Pass output channels
    .DATA_WIDTH(DATA_WIDTH),
    .FRAC_WIDTH(FRAC_WIDTH)
) conv_block_inst (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_conv),
    .feature_in(feature_in),
    .feature_valid(feature_valid && (state == CONV)),
    .gamma(gamma),
    .beta(beta),
    .mean(mean),
    .variance(variance),
    .feature_out(conv_out),
    .feature_valid_out(conv_valid),
    .conv_block_done(conv_done)
);

// MaxPool2D Layer
maxpool2d #(
    .IN_HEIGHT(IMG_HEIGHT),
    .IN_WIDTH(IMG_WIDTH),
    .CHANNELS(OUT_CHANNELS),               // ✅ Feature map channels after conv_block (FIXED: Added missing comma)
    .POOL_SIZE(2),
    .STRIDE(2),
    .DATA_WIDTH(DATA_WIDTH)
) maxpool_inst (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_pool),
    .feature_in(conv_out),
    .feature_valid(conv_valid && (state == POOL)),
    .feature_out(pooled_out),
    .feature_valid_out(pooled_valid),
    .maxpool_done(pool_done)
);

// Output assignments
assign skip_out = conv_out;
assign skip_valid = conv_valid && (state == CONV);
assign encoder_done = encoder_done_reg;

// Control FSM
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        start_conv <= 0;
        start_pool <= 0;
        encoder_done_reg <= 0;
        $display("DEBUG: Encoder Block Reset - Size: %0dx%0dx%0d->%0d", IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS, OUT_CHANNELS);
    end else begin
        case (state)
            IDLE: begin
                encoder_done_reg <= 0;
                if (start) begin
                    state <= CONV;
                    start_conv <= 1;
                    $display("DEBUG: Encoder Block started - CONV phase");
                end
            end
            
            CONV: begin
                start_conv <= 0;
                if (conv_done) begin
                    state <= POOL;
                    start_pool <= 1;
                    $display("DEBUG: Encoder Block CONV->POOL transition");
                end
            end
            
            POOL: begin
                start_pool <= 0;
                if (pool_done) begin
                    state <= DONE;
                    encoder_done_reg <= 1;
                    $display("DEBUG: Encoder Block completed");
                end
            end
            
            DONE: begin
                encoder_done_reg <= 1;
            end
            
            default: begin
                state <= IDLE;
            end
        endcase
    end
end

// Debug: Monitor state changes
reg [1:0] prev_state = IDLE;
always @(posedge clk) begin
    if (state != prev_state) begin
        $display("DEBUG: Encoder Block State change to %s at time %0t", 
                 (state == IDLE) ? "IDLE" :
                 (state == CONV) ? "CONV" :
                 (state == POOL) ? "POOL" :
                 (state == DONE) ? "DONE" : "UNKNOWN", $time);
        prev_state <= state;
    end
end

endmodule
