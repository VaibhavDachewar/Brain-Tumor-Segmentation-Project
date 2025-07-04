`timescale 1ns/1ps

module unet_top #(
    parameter IMG_HEIGHT = 256,
    parameter IMG_WIDTH = 256,
    parameter IN_CHANNELS = 3,
    parameter DATA_WIDTH = 16,
    parameter FRAC_WIDTH = 8
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    
    // Input image data
    input wire [DATA_WIDTH-1:0] image_in,
    input wire image_valid,
    
    // BatchNorm parameters (simplified - would typically be arrays)
    input wire [DATA_WIDTH-1:0] gamma,
    input wire [DATA_WIDTH-1:0] beta,
    input wire [DATA_WIDTH-1:0] mean,
    input wire [DATA_WIDTH-1:0] variance,
    
    // Convolution weights (simplified)
    input wire [DATA_WIDTH-1:0] kernel_weight,
    input wire [DATA_WIDTH-1:0] bias,
    
    // Output
    output wire [DATA_WIDTH-1:0] segmentation_out,
    output wire segmentation_valid,
    output wire unet_done
);

// Skip connection signals
wire [DATA_WIDTH-1:0] skip1_out, skip2_out, skip3_out, skip4_out;
wire skip1_valid, skip2_valid, skip3_valid, skip4_valid;

// Encoder outputs (pooled features)
wire [DATA_WIDTH-1:0] enc1_out, enc2_out, enc3_out, enc4_out;
wire enc1_valid, enc2_valid, enc3_valid, enc4_valid;
wire enc1_done, enc2_done, enc3_done, enc4_done;

// Bottleneck signals
wire [DATA_WIDTH-1:0] bottleneck_out;
wire bottleneck_valid, bottleneck_done;

// Decoder outputs
wire [DATA_WIDTH-1:0] dec1_out, dec2_out, dec3_out, dec4_out;
wire dec1_valid, dec2_valid, dec3_valid, dec4_valid;
wire dec1_done, dec2_done, dec3_done, dec4_done;

// Final convolution signals
wire [DATA_WIDTH-1:0] final_conv_out;
wire final_conv_valid, final_conv_done;

// State machine for orchestrating U-Net
reg [3:0] state;
localparam IDLE = 4'b0000;
localparam ENC1 = 4'b0001;
localparam ENC2 = 4'b0010;
localparam ENC3 = 4'b0011;
localparam ENC4 = 4'b0100;
localparam BOTTLENECK = 4'b0101;
localparam DEC1 = 4'b0110;
localparam DEC2 = 4'b0111;
localparam DEC3 = 4'b1000;
localparam DEC4 = 4'b1001;
localparam FINAL_CONV = 4'b1010;
localparam DONE = 4'b1011;

reg start_enc1, start_enc2, start_enc3, start_enc4;
reg start_bottleneck, start_dec1, start_dec2, start_dec3, start_dec4;
reg start_final_conv;
reg unet_done_reg;

// ===== ENCODER PATH =====

// Encoder Block 1: 256x256x3 -> 256x256x64 (skip) + 128x128x64 (pooled)
encoder_block #(
    .IMG_HEIGHT(256), .IMG_WIDTH(256), .CHANNELS(64),
    .DATA_WIDTH(DATA_WIDTH), .FRAC_WIDTH(FRAC_WIDTH)
) encoder1 (
    .clk(clk), .rst_n(rst_n), .start(start_enc1),
    .feature_in(image_in), .feature_valid(image_valid && (state == ENC1)),
    .gamma(gamma), .beta(beta), .mean(mean), .variance(variance),
    .skip_out(skip1_out), .skip_valid(skip1_valid),
    .pooled_out(enc1_out), .pooled_valid(enc1_valid),
    .encoder_done(enc1_done)
);

// Encoder Block 2: 128x128x64 -> 128x128x128 (skip) + 64x64x128 (pooled)
encoder_block #(
    .IMG_HEIGHT(128), .IMG_WIDTH(128), .CHANNELS(128),
    .DATA_WIDTH(DATA_WIDTH), .FRAC_WIDTH(FRAC_WIDTH)
) encoder2 (
    .clk(clk), .rst_n(rst_n), .start(start_enc2),
    .feature_in(enc1_out), .feature_valid(enc1_valid && (state == ENC2)),
    .gamma(gamma), .beta(beta), .mean(mean), .variance(variance),
    .skip_out(skip2_out), .skip_valid(skip2_valid),
    .pooled_out(enc2_out), .pooled_valid(enc2_valid),
    .encoder_done(enc2_done)
);

// Encoder Block 3: 64x64x128 -> 64x64x256 (skip) + 32x32x256 (pooled)
encoder_block #(
    .IMG_HEIGHT(64), .IMG_WIDTH(64), .CHANNELS(256),
    .DATA_WIDTH(DATA_WIDTH), .FRAC_WIDTH(FRAC_WIDTH)
) encoder3 (
    .clk(clk), .rst_n(rst_n), .start(start_enc3),
    .feature_in(enc2_out), .feature_valid(enc2_valid && (state == ENC3)),
    .gamma(gamma), .beta(beta), .mean(mean), .variance(variance),
    .skip_out(skip3_out), .skip_valid(skip3_valid),
    .pooled_out(enc3_out), .pooled_valid(enc3_valid),
    .encoder_done(enc3_done)
);

// Encoder Block 4: 32x32x256 -> 32x32x512 (skip) + 16x16x512 (pooled)
encoder_block #(
    .IMG_HEIGHT(32), .IMG_WIDTH(32), .CHANNELS(512),
    .DATA_WIDTH(DATA_WIDTH), .FRAC_WIDTH(FRAC_WIDTH)
) encoder4 (
    .clk(clk), .rst_n(rst_n), .start(start_enc4),
    .feature_in(enc3_out), .feature_valid(enc3_valid && (state == ENC4)),
    .gamma(gamma), .beta(beta), .mean(mean), .variance(variance),
    .skip_out(skip4_out), .skip_valid(skip4_valid),
    .pooled_out(enc4_out), .pooled_valid(enc4_valid),
    .encoder_done(enc4_done)
);

// ===== BOTTLENECK =====

// Bottleneck: 16x16x512 -> 16x16x1024
conv_block #(
    .IMG_HEIGHT(16), .IMG_WIDTH(16), .CHANNELS(1024),
    .DATA_WIDTH(DATA_WIDTH), .FRAC_WIDTH(FRAC_WIDTH)
) bottleneck (
    .clk(clk), .rst_n(rst_n), .start(start_bottleneck),
    .feature_in(enc4_out), .feature_valid(enc4_valid && (state == BOTTLENECK)),
    .gamma(gamma), .beta(beta), .mean(mean), .variance(variance),
    .feature_out(bottleneck_out), .feature_valid_out(bottleneck_valid),
    .conv_block_done(bottleneck_done)
);

// ===== DECODER PATH =====

// Decoder Block 1: 16x16x1024 -> 32x32x512 (with skip4: 32x32x512)
decoder_block #(
    .IN_HEIGHT(16), .IN_WIDTH(16), .IN_CHANNELS(1024),
    .SKIP_CHANNELS(512), .OUT_CHANNELS(512),
    .DATA_WIDTH(DATA_WIDTH), .FRAC_WIDTH(FRAC_WIDTH)
) decoder1 (
    .clk(clk), .rst_n(rst_n), .start(start_dec1),
    .feature_in(bottleneck_out), .feature_valid(bottleneck_valid && (state == DEC1)),
    .skip_in(skip4_out), .skip_valid(skip4_valid && (state == DEC1)),
    .gamma(gamma), .beta(beta), .mean(mean), .variance(variance),
    .kernel_weight(kernel_weight), .bias(bias),
    .feature_out(dec1_out), .feature_valid_out(dec1_valid),
    .decoder_done(dec1_done)
);

// Decoder Block 2: 32x32x512 -> 64x64x256 (with skip3: 64x64x256)
decoder_block #(
    .IN_HEIGHT(32), .IN_WIDTH(32), .IN_CHANNELS(512),
    .SKIP_CHANNELS(256), .OUT_CHANNELS(256),
    .DATA_WIDTH(DATA_WIDTH), .FRAC_WIDTH(FRAC_WIDTH)
) decoder2 (
    .clk(clk), .rst_n(rst_n), .start(start_dec2),
    .feature_in(dec1_out), .feature_valid(dec1_valid && (state == DEC2)),
    .skip_in(skip3_out), .skip_valid(skip3_valid && (state == DEC2)),
    .gamma(gamma), .beta(beta), .mean(mean), .variance(variance),
    .kernel_weight(kernel_weight), .bias(bias),
    .feature_out(dec2_out), .feature_valid_out(dec2_valid),
    .decoder_done(dec2_done)
);

// Decoder Block 3: 64x64x256 -> 128x128x128 (with skip2: 128x128x128)
decoder_block #(
    .IN_HEIGHT(64), .IN_WIDTH(64), .IN_CHANNELS(256),
    .SKIP_CHANNELS(128), .OUT_CHANNELS(128),
    .DATA_WIDTH(DATA_WIDTH), .FRAC_WIDTH(FRAC_WIDTH)
) decoder3 (
    .clk(clk), .rst_n(rst_n), .start(start_dec3),
    .feature_in(dec2_out), .feature_valid(dec2_valid && (state == DEC3)),
    .skip_in(skip2_out), .skip_valid(skip2_valid && (state == DEC3)),
    .gamma(gamma), .beta(beta), .mean(mean), .variance(variance),
    .kernel_weight(kernel_weight), .bias(bias),
    .feature_out(dec3_out), .feature_valid_out(dec3_valid),
    .decoder_done(dec3_done)
);

// Decoder Block 4: 128x128x128 -> 256x256x64 (with skip1: 256x256x64)
decoder_block #(
    .IN_HEIGHT(128), .IN_WIDTH(128), .IN_CHANNELS(128),
    .SKIP_CHANNELS(64), .OUT_CHANNELS(64),
    .DATA_WIDTH(DATA_WIDTH), .FRAC_WIDTH(FRAC_WIDTH)
) decoder4 (
    .clk(clk), .rst_n(rst_n), .start(start_dec4),
    .feature_in(dec3_out), .feature_valid(dec3_valid && (state == DEC4)),
    .skip_in(skip1_out), .skip_valid(skip1_valid && (state == DEC4)),
    .gamma(gamma), .beta(beta), .mean(mean), .variance(variance),
    .kernel_weight(kernel_weight), .bias(bias),
    .feature_out(dec4_out), .feature_valid_out(dec4_valid),
    .decoder_done(dec4_done)
);

// ===== FINAL OUTPUT LAYER =====

// Final 1x1 Conv2D with Sigmoid: 256x256x64 -> 256x256x1
conv2d_layer #(
    .IMG_HEIGHT(256),
    .IMG_WIDTH(256),
    .IN_CHANNELS(64),
    .OUT_CHANNELS(1),
    .KERNEL_SIZE(1),
    .DATA_WIDTH(DATA_WIDTH)
) final_conv (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_final_conv),
    .pixel_in(dec4_out),
    .pixel_valid(dec4_valid && (state == FINAL_CONV)),
    .feature_out(final_conv_out),
    .feature_valid(final_conv_valid),
    .conv_done(final_conv_done)
);

// Apply Sigmoid activation (simplified - you may need a dedicated sigmoid module)
reg [DATA_WIDTH-1:0] sigmoid_out;
reg sigmoid_valid;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        sigmoid_out <= 0;
        sigmoid_valid <= 0;
    end else begin
        // Simplified sigmoid approximation (you should implement proper sigmoid)
        // For now, just pass through the final conv output
        sigmoid_out <= final_conv_out;
        sigmoid_valid <= final_conv_valid;
    end
end

// Output assignments
assign segmentation_out = sigmoid_out;
assign segmentation_valid = sigmoid_valid && (state == FINAL_CONV);
assign unet_done = unet_done_reg;

// ===== CONTROL FSM =====
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        start_enc1 <= 0;
        start_enc2 <= 0;
        start_enc3 <= 0;
        start_enc4 <= 0;
        start_bottleneck <= 0;
        start_dec1 <= 0;
        start_dec2 <= 0;
        start_dec3 <= 0;
        start_dec4 <= 0;
        start_final_conv <= 0;
        unet_done_reg <= 0;
        $display("DEBUG: U-Net Top Reset - Input: %0dx%0dx%0d", IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS);
    end else begin
        case (state)
            IDLE: begin
                unet_done_reg <= 0;
                if (start) begin
                    state <= ENC1;
                    start_enc1 <= 1;
                    $display("DEBUG: U-Net started - ENC1 phase at time %0t", $time);
                end
            end
            
            ENC1: begin
                start_enc1 <= 0;
                if (enc1_done) begin
                    state <= ENC2;
                    start_enc2 <= 1;
                    $display("DEBUG: U-Net ENC1->ENC2 transition at time %0t", $time);
                end
            end
            
            ENC2: begin
                start_enc2 <= 0;
                if (enc2_done) begin
                    state <= ENC3;
                    start_enc3 <= 1;
                    $display("DEBUG: U-Net ENC2->ENC3 transition at time %0t", $time);
                end
            end
            
            ENC3: begin
                start_enc3 <= 0;
                if (enc3_done) begin
                    state <= ENC4;
                    start_enc4 <= 1;
                    $display("DEBUG: U-Net ENC3->ENC4 transition at time %0t", $time);
                end
            end
            
            ENC4: begin
                start_enc4 <= 0;
                if (enc4_done) begin
                    state <= BOTTLENECK;
                    start_bottleneck <= 1;
                    $display("DEBUG: U-Net ENC4->BOTTLENECK transition at time %0t", $time);
                end
            end
            
            BOTTLENECK: begin
                start_bottleneck <= 0;
                if (bottleneck_done) begin
                    state <= DEC1;
                    start_dec1 <= 1;
                    $display("DEBUG: U-Net BOTTLENECK->DEC1 transition at time %0t", $time);
                end
            end
            
            DEC1: begin
                start_dec1 <= 0;
                if (dec1_done) begin
                    state <= DEC2;
                    start_dec2 <= 1;
                    $display("DEBUG: U-Net DEC1->DEC2 transition at time %0t", $time);
                end
            end
            
            DEC2: begin
                start_dec2 <= 0;
                if (dec2_done) begin
                    state <= DEC3;
                    start_dec3 <= 1;
                    $display("DEBUG: U-Net DEC2->DEC3 transition at time %0t", $time);
                end
            end
            
            DEC3: begin
                start_dec3 <= 0;
                if (dec3_done) begin
                    state <= DEC4;
                    start_dec4 <= 1;
                    $display("DEBUG: U-Net DEC3->DEC4 transition at time %0t", $time);
                end
            end
            
            DEC4: begin
                start_dec4 <= 0;
                if (dec4_done) begin
                    state <= FINAL_CONV;
                    start_final_conv <= 1;
                    $display("DEBUG: U-Net DEC4->FINAL_CONV transition at time %0t", $time);
                end
            end
            
            FINAL_CONV: begin
                start_final_conv <= 0;
                if (final_conv_done) begin
                    state <= DONE;
                    unet_done_reg <= 1;
                    $display("DEBUG: U-Net completed at time %0t", $time);
                end
            end
            
            DONE: begin
                unet_done_reg <= 1;
                // Stay in DONE state until reset or new start
            end
            
            default: begin
                state <= IDLE;
            end
        endcase
    end
end

// Debug: Monitor major state transitions
reg [3:0] prev_state = IDLE;
always @(posedge clk) begin
    if (state != prev_state) begin
        $display("DEBUG: U-Net State change from %s to %s at time %0t", 
                 get_state_name(prev_state), get_state_name(state), $time);
        prev_state <= state;
    end
end

// Function to get state name for debugging
function [127:0] get_state_name;
    input [3:0] state_value;
    begin
        case (state_value)
            IDLE: get_state_name = "IDLE";
            ENC1: get_state_name = "ENC1";
            ENC2: get_state_name = "ENC2";
            ENC3: get_state_name = "ENC3";
            ENC4: get_state_name = "ENC4";
            BOTTLENECK: get_state_name = "BOTTLENECK";
            DEC1: get_state_name = "DEC1";
            DEC2: get_state_name = "DEC2";
            DEC3: get_state_name = "DEC3";
            DEC4: get_state_name = "DEC4";
            FINAL_CONV: get_state_name = "FINAL_CONV";
            DONE: get_state_name = "DONE";
            default: get_state_name = "UNKNOWN";
        endcase
    end
endfunction

endmodule
