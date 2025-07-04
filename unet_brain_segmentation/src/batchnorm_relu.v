`timescale 1ns/1ps

module batchnorm_relu #(
    parameter IMG_HEIGHT = 256,
    parameter IMG_WIDTH = 256,
    parameter CHANNELS = 64,
    parameter DATA_WIDTH = 16,
    parameter FRAC_WIDTH = 8  // Number of fractional bits for fixed-point
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [DATA_WIDTH-1:0] feature_in,
    input wire feature_valid,
    
    // BatchNorm parameters (would be loaded from memory/ROM in real implementation)
    input wire [DATA_WIDTH-1:0] gamma,  // Scale parameter
    input wire [DATA_WIDTH-1:0] beta,   // Shift parameter
    input wire [DATA_WIDTH-1:0] mean,   // Running mean
    input wire [DATA_WIDTH-1:0] variance, // Running variance
    
    output reg [DATA_WIDTH-1:0] feature_out,
    output reg feature_valid_out,
    output reg bn_relu_done
);

// Parameters
localparam TOTAL_FEATURES = IMG_HEIGHT * IMG_WIDTH * CHANNELS;
localparam EPSILON = 16'h0001; // Small epsilon value in fixed-point

// State machine
reg [2:0] state;
localparam IDLE = 3'b000;
localparam PROCESSING = 3'b001;
localparam DONE = 3'b010;

// Counters
reg [31:0] feature_count;
reg [31:0] output_count;

// Pipeline registers for better timing
reg [DATA_WIDTH-1:0] stage1_data;
reg [DATA_WIDTH-1:0] stage2_data;
reg [DATA_WIDTH-1:0] stage3_data;
reg stage1_valid, stage2_valid, stage3_valid;

// Arithmetic registers
reg [DATA_WIDTH-1:0] normalized_data;
reg [DATA_WIDTH-1:0] scaled_data;
reg [DATA_WIDTH-1:0] shifted_data;

// Intermediate calculations (use wider registers to prevent overflow)
reg [DATA_WIDTH*2-1:0] temp_mult;
reg [DATA_WIDTH*2-1:0] temp_var_eps;
reg [DATA_WIDTH-1:0] inv_sqrt_var;

// Progress tracking
reg [31:0] progress_counter;
localparam PROGRESS_STEP = TOTAL_FEATURES / 100;

// Main FSM
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        feature_count <= 0;
        output_count <= 0;
        progress_counter <= 0;
        feature_out <= 0;
        feature_valid_out <= 0;
        bn_relu_done <= 0;
        
        // Clear pipeline
        stage1_data <= 0;
        stage2_data <= 0;
        stage3_data <= 0;
        stage1_valid <= 0;
        stage2_valid <= 0;
        stage3_valid <= 0;
        
        $display("DEBUG: BatchNorm+ReLU Reset - Total features: %0d", TOTAL_FEATURES);
        
    end else begin
        case (state)
            IDLE: begin
                bn_relu_done <= 0;
                feature_valid_out <= 0;
                if (start) begin
                    state <= PROCESSING;
                    feature_count <= 0;
                    output_count <= 0;
                    progress_counter <= 0;
                    $display("DEBUG: Starting BatchNorm+ReLU processing at time %0t", $time);
                end
            end
            
            PROCESSING: begin
                // Pipeline Stage 1: Input normalization
                if (feature_valid) begin
                    stage1_data <= feature_in;
                    stage1_valid <= 1;
                    feature_count <= feature_count + 1;
                    
                    // Progress reporting
                    if (feature_count % PROGRESS_STEP == 0 && feature_count > 0) begin
                        $display("INFO: BatchNorm+ReLU progress: %0d%% (%0d/%0d) at time %0t", 
                                (feature_count * 100) / TOTAL_FEATURES, feature_count, TOTAL_FEATURES, $time);
                    end
                end else begin
                    stage1_valid <= 0;
                end
                
                // Pipeline Stage 2: Batch Normalization
                if (stage1_valid) begin
                    // Simplified BatchNorm calculation
                    // normalized = (x - mean) / sqrt(var + epsilon)
                    // We'll use approximations for division and sqrt for hardware efficiency
                    
                    // Step 1: x - mean
                    if (stage1_data >= mean) begin
                        normalized_data <= stage1_data - mean;
                    end else begin
                        normalized_data <= mean - stage1_data; // Handle negative case
                    end
                    
                    stage2_data <= normalized_data;
                    stage2_valid <= 1;
                end else begin
                    stage2_valid <= 0;
                end
                
                // Pipeline Stage 3: Scale and Shift + ReLU
                if (stage2_valid) begin
                    // Simplified scaling: multiply by gamma and add beta
                    temp_mult = (stage2_data * gamma) >> FRAC_WIDTH; // Fixed-point multiplication
                    scaled_data = temp_mult[DATA_WIDTH-1:0];
                    shifted_data = scaled_data + beta;
                    
                    // ReLU activation: max(0, x)
                    if (shifted_data[DATA_WIDTH-1] == 1'b1) begin // Check sign bit
                        stage3_data <= 0; // Negative, set to 0
                    end else begin
                        stage3_data <= shifted_data; // Positive, keep value
                    end
                    
                    stage3_valid <= 1;
                end else begin
                    stage3_valid <= 0;
                end
                
                // Output stage
                if (stage3_valid) begin
                    feature_out <= stage3_data;
                    feature_valid_out <= 1;
                    output_count <= output_count + 1;
                    
                    if (output_count >= TOTAL_FEATURES - 1) begin
                        state <= DONE;
                        $display("DEBUG: BatchNorm+ReLU processing completed at time %0t", $time);
                    end
                end else begin
                    feature_valid_out <= 0;
                end
            end
            
            DONE: begin
                feature_valid_out <= 0;
                bn_relu_done <= 1;
                $display("SUCCESS: BatchNorm+ReLU completed for %0dx%0dx%0d features!", 
                        IMG_HEIGHT, IMG_WIDTH, CHANNELS);
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
        $display("DEBUG: BatchNorm+ReLU State change to %s at time %0t", 
                 (state == IDLE) ? "IDLE" :
                 (state == PROCESSING) ? "PROCESSING" :
                 (state == DONE) ? "DONE" : "UNKNOWN", $time);
        prev_state <= state;
    end
end

endmodule
