`timescale 1ns/1ps

module conv_block #(
    parameter IMG_HEIGHT = 256,
    parameter IMG_WIDTH = 256,
    parameter IN_CHANNELS = 3,         // ✅ for first conv layer
    parameter OUT_CHANNELS = 64,       // ✅ output of conv layer
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
    
    output wire [DATA_WIDTH-1:0] feature_out,
    output wire feature_valid_out,
    output wire conv_block_done
);

// Internal signals
wire [DATA_WIDTH-1:0] conv1_out, bn1_out, conv2_out, bn2_out;
wire conv1_valid, bn1_valid, conv2_valid, bn2_valid;
wire conv1_done, bn1_done, conv2_done, bn2_done;

// State machine for orchestrating the sequence
reg [2:0] state;
localparam IDLE = 3'b000;
localparam CONV1 = 3'b001;
localparam BN1 = 3'b010;
localparam CONV2 = 3'b011;
localparam BN2 = 3'b100;
localparam DONE = 3'b101;

reg start_conv1, start_bn1, start_conv2, start_bn2;
reg conv_block_done_reg;

// First Conv2D Layer
conv2d_layer #(
    .IMG_HEIGHT(IMG_HEIGHT),
    .IMG_WIDTH(IMG_WIDTH),
    .IN_CHANNELS(IN_CHANNELS),         // ✅ actual input channels (e.g., 3)
    .OUT_CHANNELS(OUT_CHANNELS),
    .KERNEL_SIZE(3),
    .DATA_WIDTH(DATA_WIDTH)
) conv2d_1  (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_conv1),
    .pixel_in(feature_in),
    .pixel_valid(feature_valid && (state == CONV1)),
    .feature_out(conv1_out),
    .feature_valid(conv1_valid),
    .conv_done(conv1_done)
);

// First BatchNorm + ReLU
batchnorm_relu #(
    .IMG_HEIGHT(IMG_HEIGHT),
    .IMG_WIDTH(IMG_WIDTH),
    .CHANNELS(OUT_CHANNELS),
    .DATA_WIDTH(DATA_WIDTH),
    .FRAC_WIDTH(FRAC_WIDTH)
) bn_relu_1 (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_bn1),
    .feature_in(conv1_out),
    .feature_valid(conv1_valid && (state == BN1)),
    .gamma(gamma),
    .beta(beta),
    .mean(mean),
    .variance(variance),
    .feature_out(bn1_out),
    .feature_valid_out(bn1_valid),
    .bn_relu_done(bn1_done)
);

// Second Conv2D Layer
// Second Conv2D Layer
conv2d_layer #(
    .IMG_HEIGHT(IMG_HEIGHT),
    .IMG_WIDTH(IMG_WIDTH),
    .IN_CHANNELS(OUT_CHANNELS),        // ✅ output of conv1 is input here
    .OUT_CHANNELS(OUT_CHANNELS),
    .KERNEL_SIZE(3),
    .DATA_WIDTH(DATA_WIDTH)
) conv2d_2 (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_conv2),
    .pixel_in(bn1_out),
    .pixel_valid(bn1_valid && (state == CONV2)),
    .feature_out(conv2_out),
    .feature_valid(conv2_valid),
    .conv_done(conv2_done)
);

// Second BatchNorm + ReLU
batchnorm_relu #(
    .IMG_HEIGHT(IMG_HEIGHT),
    .IMG_WIDTH(IMG_WIDTH),
    .CHANNELS(OUT_CHANNELS),
    .DATA_WIDTH(DATA_WIDTH),
    .FRAC_WIDTH(FRAC_WIDTH)
) bn_relu_2 (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_bn2),
    .feature_in(conv2_out),
    .feature_valid(conv2_valid && (state == BN2)),
    .gamma(gamma),
    .beta(beta),
    .mean(mean),
    .variance(variance),
    .feature_out(bn2_out),
    .feature_valid_out(bn2_valid),
    .bn_relu_done(bn2_done)
);

// Output assignment
assign feature_out = bn2_out;
assign feature_valid_out = bn2_valid && (state == BN2);
assign conv_block_done = conv_block_done_reg;

// Control FSM
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        start_conv1 <= 0;
        start_bn1 <= 0;
        start_conv2 <= 0;
        start_bn2 <= 0;
        conv_block_done_reg <= 0;
        $display("DEBUG: Conv Block Reset");
    end else begin
        case (state)
            IDLE: begin
                conv_block_done_reg <= 0;
                if (start) begin
                    state <= CONV1;
                    start_conv1 <= 1;
                    $display("DEBUG: Conv Block started - CONV1 phase");
                end
            end
            
            CONV1: begin
                start_conv1 <= 0;
                if (conv1_done) begin
                    state <= BN1;
                    start_bn1 <= 1;
                    $display("DEBUG: Conv Block CONV1->BN1 transition");
                end
            end
            
            BN1: begin
                start_bn1 <= 0;
                if (bn1_done) begin
                    state <= CONV2;
                    start_conv2 <= 1;
                    $display("DEBUG: Conv Block BN1->CONV2 transition");
                end
            end
            
            CONV2: begin
                start_conv2 <= 0;
                if (conv2_done) begin
                    state <= BN2;
                    start_bn2 <= 1;
                    $display("DEBUG: Conv Block CONV2->BN2 transition");
                end
            end
            
            BN2: begin
                start_bn2 <= 0;
                if (bn2_done) begin
                    state <= DONE;
                    conv_block_done_reg <= 1;
                    $display("DEBUG: Conv Block completed");
                end
            end
            
            DONE: begin
                conv_block_done_reg <= 1;
            end
            
            default: begin
                state <= IDLE;
            end
        endcase
    end
end

endmodule
