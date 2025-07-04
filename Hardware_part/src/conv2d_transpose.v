`timescale 1ns/1ps

module conv2d_transpose #(
    parameter IN_HEIGHT = 16,        // Input height (from bottleneck)
    parameter IN_WIDTH = 16,         // Input width (from bottleneck)
    parameter IN_CHANNELS = 1024,    // Input channels (from bottleneck)
    parameter OUT_CHANNELS = 512,    // Output channels (for decoder)
    parameter KERNEL_SIZE = 2,       // Transpose kernel size
    parameter STRIDE = 2,            // Stride for upsampling
    parameter DATA_WIDTH = 16,       // Data width
    parameter FRAC_WIDTH = 8         // Fractional bits for fixed-point
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [DATA_WIDTH-1:0] feature_in,
    input wire feature_valid,
    
    // Kernel weights (simplified - in real implementation would be loaded from memory)
    input wire [DATA_WIDTH-1:0] kernel_weight,
    input wire [DATA_WIDTH-1:0] bias,
    
    output reg [DATA_WIDTH-1:0] feature_out,
    output reg feature_valid_out,
    output reg transpose_done
);

// Calculate output dimensions
localparam OUT_HEIGHT = IN_HEIGHT * STRIDE;  // 16 * 2 = 32
localparam OUT_WIDTH = IN_WIDTH * STRIDE;    // 16 * 2 = 32
localparam TOTAL_INPUT_FEATURES = IN_HEIGHT * IN_WIDTH * IN_CHANNELS;
localparam TOTAL_OUTPUT_FEATURES = OUT_HEIGHT * OUT_WIDTH * OUT_CHANNELS;

// Memory for input and output features
reg [DATA_WIDTH-1:0] input_buffer [0:TOTAL_INPUT_FEATURES-1];
reg [DATA_WIDTH-1:0] output_buffer [0:TOTAL_OUTPUT_FEATURES-1];

// State machine
reg [2:0] state;
localparam IDLE = 3'b000;
localparam LOAD_INPUT = 3'b001;
localparam PROCESS = 3'b010;
localparam OUTPUT = 3'b011;
localparam DONE = 3'b100;

// Counters and indices
reg [31:0] input_count;
reg [31:0] output_count;
reg [31:0] process_count;

// Processing coordinates
reg [31:0] out_ch, out_row, out_col;
reg [31:0] in_ch, in_row, in_col;
reg [31:0] k_row, k_col;

// Intermediate calculation registers
reg [DATA_WIDTH*2-1:0] accumulator;
reg [DATA_WIDTH-1:0] temp_result;

// Progress tracking
reg [31:0] progress_counter;
localparam PROGRESS_STEP = TOTAL_INPUT_FEATURES / 100;

// Helper function to get input buffer index
function [31:0] get_input_index;
    input [31:0] row, col, channel;
    begin
        get_input_index = (channel * IN_HEIGHT * IN_WIDTH) + (row * IN_WIDTH) + col;
    end
endfunction

// Helper function to get output buffer index
function [31:0] get_output_index;
    input [31:0] row, col, channel;
    begin
        get_output_index = (channel * OUT_HEIGHT * OUT_WIDTH) + (row * OUT_WIDTH) + col;
    end
endfunction

// Main FSM
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        input_count <= 0;
        output_count <= 0;
        process_count <= 0;
        feature_out <= 0;
        feature_valid_out <= 0;
        transpose_done <= 0;
        
        // Initialize coordinates
        out_ch <= 0;
        out_row <= 0;
        out_col <= 0;
        in_ch <= 0;
        in_row <= 0;
        in_col <= 0;
        k_row <= 0;
        k_col <= 0;
        
        accumulator <= 0;
        progress_counter <= 0;
        
        $display("DEBUG: Conv2DTranspose Reset");
        $display("Input size: %0dx%0dx%0d", IN_HEIGHT, IN_WIDTH, IN_CHANNELS);
        $display("Output size: %0dx%0dx%0d", OUT_HEIGHT, OUT_WIDTH, OUT_CHANNELS);
        $display("Kernel size: %0dx%0d, Stride: %0d", KERNEL_SIZE, KERNEL_SIZE, STRIDE);
        $display("Total input features: %0d", TOTAL_INPUT_FEATURES);
        $display("Total output features: %0d", TOTAL_OUTPUT_FEATURES);
        
    end else begin
        case (state)
            IDLE: begin
                transpose_done <= 0;
                feature_valid_out <= 0;
                if (start) begin
                    state <= LOAD_INPUT;
                    input_count <= 0;
                    progress_counter <= 0;
                    $display("DEBUG: Starting Conv2DTranspose at time %0t", $time);
                end
            end
            
            LOAD_INPUT: begin
                if (feature_valid) begin
                    input_buffer[input_count] <= feature_in;
                    
                    // Progress reporting
                    if (input_count % PROGRESS_STEP == 0 && input_count > 0) begin
                        $display("INFO: Conv2DTranspose loading progress: %0d%% (%0d/%0d) at time %0t", 
                                (input_count * 100) / TOTAL_INPUT_FEATURES, 
                                input_count, TOTAL_INPUT_FEATURES, $time);
                    end
                    
                    input_count <= input_count + 1;
                    
                    if (input_count == TOTAL_INPUT_FEATURES - 1) begin
                        state <= PROCESS;
                        process_count <= 0;
                        out_ch <= 0;
                        out_row <= 0;
                        out_col <= 0;
                        $display("DEBUG: Loading complete, starting processing at time %0t", $time);
                        
                        // Initialize output buffer to zero
                        for (integer i = 0; i < TOTAL_OUTPUT_FEATURES; i = i + 1) begin
                            output_buffer[i] <= 0;
                        end
                    end
                end
            end
            
            PROCESS: begin
                // Simplified transposed convolution processing
                // For each output position, accumulate contributions from input
                if (process_count < TOTAL_OUTPUT_FEATURES) begin
                    
                    // Initialize accumulator for this output position
                    accumulator <= 0;
                    
                    // Calculate which input positions contribute to this output
                    // For stride=2, kernel=2: each input pixel contributes to a 2x2 region in output
                    
                    // Find the input position that contributes to current output
                    in_row = out_row / STRIDE;
                    in_col = out_col / STRIDE;
                    
                    // Check if this output position receives contribution from input
                    if ((out_row % STRIDE < KERNEL_SIZE) && (out_col % STRIDE < KERNEL_SIZE) && 
                        (in_row < IN_HEIGHT) && (in_col < IN_WIDTH)) begin
                        
                        // Accumulate from all input channels to current output channel
                        for (in_ch = 0; in_ch < IN_CHANNELS; in_ch = in_ch + 1) begin
                            accumulator <= accumulator + 
                                (input_buffer[get_input_index(in_row, in_col, in_ch)] * kernel_weight);
                        end
                        
                        // Add bias and store result
                        temp_result = (accumulator >> FRAC_WIDTH) + bias;
                        output_buffer[get_output_index(out_row, out_col, out_ch)] <= temp_result;
                    end
                    
                    process_count <= process_count + 1;
                    
                    // Progress reporting
                    if (process_count % (PROGRESS_STEP/4) == 0 && process_count > 0) begin
                        $display("INFO: Conv2DTranspose processing progress: %0d%% (%0d/%0d) at time %0t", 
                                (process_count * 100) / TOTAL_OUTPUT_FEATURES, 
                                process_count, TOTAL_OUTPUT_FEATURES, $time);
                    end
                    
                    // Update output coordinates
                    if (out_col < OUT_WIDTH - 1) begin
                        out_col <= out_col + 1;
                    end else begin
                        out_col <= 0;
                        if (out_row < OUT_HEIGHT - 1) begin
                            out_row <= out_row + 1;
                        end else begin
                            out_row <= 0;
                            if (out_ch < OUT_CHANNELS - 1) begin
                                out_ch <= out_ch + 1;
                            end else begin
                                // Processing complete
                                state <= OUTPUT;
                                output_count <= 0;
                                $display("DEBUG: Processing complete, starting output at time %0t", $time);
                            end
                        end
                    end
                end
            end
            
            OUTPUT: begin
                // Output processed features
                if (output_count < TOTAL_OUTPUT_FEATURES) begin
                    feature_out <= output_buffer[output_count];
                    feature_valid_out <= 1;
                    output_count <= output_count + 1;
                    
                    if (output_count == TOTAL_OUTPUT_FEATURES - 1) begin
                        state <= DONE;
                        $display("DEBUG: Output complete at time %0t", $time);
                    end
                end else begin
                    feature_valid_out <= 0;
                end
            end
            
            DONE: begin
                feature_valid_out <= 0;
                transpose_done <= 1;
                $display("SUCCESS: Conv2DTranspose completed!");
                $display("Upsampled from %0dx%0dx%0d to %0dx%0dx%0d", 
                        IN_HEIGHT, IN_WIDTH, IN_CHANNELS, OUT_HEIGHT, OUT_WIDTH, OUT_CHANNELS);
            end
            
            default: begin
                state <= IDLE;
                $display("DEBUG: Unknown state, returning to IDLE");
            end
        endcase
    end
end

// Debug: Monitor state changes
reg [2:0] prev_state = IDLE;
always @(posedge clk) begin
    if (state != prev_state) begin
        $display("DEBUG: Conv2DTranspose State change to %s at time %0t", 
                 (state == IDLE) ? "IDLE" :
                 (state == LOAD_INPUT) ? "LOAD_INPUT" :
                 (state == PROCESS) ? "PROCESS" :
                 (state == OUTPUT) ? "OUTPUT" :
                 (state == DONE) ? "DONE" : "UNKNOWN", $time);
        prev_state <= state;
    end
end

endmodule
