`timescale 1ns/1ps

module maxpool2d #(
    parameter IN_HEIGHT = 256,
    parameter IN_WIDTH = 256,
    parameter CHANNELS = 64,
    parameter POOL_SIZE = 2,
    parameter STRIDE = 2,
    parameter DATA_WIDTH = 16
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [DATA_WIDTH-1:0] feature_in,
    input wire feature_valid,
    output reg [DATA_WIDTH-1:0] feature_out,
    output reg feature_valid_out,
    output reg maxpool_done
);

// Calculate output dimensions
localparam OUT_HEIGHT = IN_HEIGHT / STRIDE;
localparam OUT_WIDTH = IN_WIDTH / STRIDE;
localparam TOTAL_INPUT_FEATURES = IN_HEIGHT * IN_WIDTH * CHANNELS;
localparam TOTAL_OUTPUT_FEATURES = OUT_HEIGHT * OUT_WIDTH * CHANNELS;

// Memory for storing input features
reg [DATA_WIDTH-1:0] input_buffer [0:TOTAL_INPUT_FEATURES-1];

// State machine
reg [2:0] state;
localparam IDLE = 3'b000;
localparam LOAD_DATA = 3'b001;
localparam PROCESS = 3'b010;
localparam DONE = 3'b011;

// Counters
reg [31:0] input_count;
reg [31:0] output_count;
reg [31:0] channel_idx;
reg [31:0] out_row, out_col;
reg [31:0] in_row, in_col;

// Pooling window registers
reg [DATA_WIDTH-1:0] pool_window [0:POOL_SIZE*POOL_SIZE-1];
reg [DATA_WIDTH-1:0] max_value;

// Progress tracking
reg [31:0] progress_counter;
localparam PROGRESS_STEP = TOTAL_INPUT_FEATURES / 100;

// Helper function to get buffer index
function [31:0] get_buffer_index;
    input [31:0] row, col, channel;
    begin
        get_buffer_index = (channel * IN_HEIGHT * IN_WIDTH) + (row * IN_WIDTH) + col;
    end
endfunction

// Helper function to find maximum
function [DATA_WIDTH-1:0] find_max;
    input [DATA_WIDTH-1:0] a, b, c, d;
    reg [DATA_WIDTH-1:0] temp_max1, temp_max2;
    begin
        temp_max1 = (a > b) ? a : b;
        temp_max2 = (c > d) ? c : d;
        find_max = (temp_max1 > temp_max2) ? temp_max1 : temp_max2;
    end
endfunction

// Main FSM
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        input_count <= 0;
        output_count <= 0;
        channel_idx <= 0;
        out_row <= 0;
        out_col <= 0;
        feature_out <= 0;
        feature_valid_out <= 0;
        maxpool_done <= 0;
        progress_counter <= 0;
        
        $display("DEBUG: MaxPool2D Reset");
        $display("Input size: %0dx%0dx%0d", IN_HEIGHT, IN_WIDTH, CHANNELS);
        $display("Output size: %0dx%0dx%0d", OUT_HEIGHT, OUT_WIDTH, CHANNELS);
        $display("Pool size: %0dx%0d, Stride: %0d", POOL_SIZE, POOL_SIZE, STRIDE);
        $display("Total input features: %0d", TOTAL_INPUT_FEATURES);
        $display("Total output features: %0d", TOTAL_OUTPUT_FEATURES);
        
    end else begin
        case (state)
            IDLE: begin
                maxpool_done <= 0;
                feature_valid_out <= 0;
                if (start) begin
                    state <= LOAD_DATA;
                    input_count <= 0;
                    output_count <= 0;
                    progress_counter <= 0;
                    $display("DEBUG: Starting MaxPool2D at time %0t", $time);
                end
            end
            
            LOAD_DATA: begin
                if (feature_valid) begin
                    input_buffer[input_count] <= feature_in;
                    
                    // Progress reporting
                    if (input_count % PROGRESS_STEP == 0 && input_count > 0) begin
                        $display("INFO: MaxPool2D loading progress: %0d%% (%0d/%0d) at time %0t", 
                                (input_count * 100) / TOTAL_INPUT_FEATURES, 
                                input_count, TOTAL_INPUT_FEATURES, $time);
                    end
                    
                    input_count <= input_count + 1;
                    
                    if (input_count == TOTAL_INPUT_FEATURES - 1) begin
                        state <= PROCESS;
                        channel_idx <= 0;
                        out_row <= 0;
                        out_col <= 0;
                        output_count <= 0;
                        $display("DEBUG: Loading complete, starting processing at time %0t", $time);
                    end
                end
            end
            
            PROCESS: begin
                // Process one output pixel per clock cycle
                if (output_count < TOTAL_OUTPUT_FEATURES) begin
                    // Calculate input coordinates for pooling window
                    in_row = out_row * STRIDE;
                    in_col = out_col * STRIDE;
                    
                    // Load pooling window (2x2)
                    pool_window[0] <= input_buffer[get_buffer_index(in_row, in_col, channel_idx)];
                    pool_window[1] <= input_buffer[get_buffer_index(in_row, in_col+1, channel_idx)];
                    pool_window[2] <= input_buffer[get_buffer_index(in_row+1, in_col, channel_idx)];
                    pool_window[3] <= input_buffer[get_buffer_index(in_row+1, in_col+1, channel_idx)];
                    
                    // Find maximum value in pooling window
                    max_value = find_max(pool_window[0], pool_window[1], pool_window[2], pool_window[3]);
                    
                    // Output the maximum value
                    feature_out <= max_value;
                    feature_valid_out <= 1;
                    output_count <= output_count + 1;
                    
                    // Progress reporting
                    if (output_count % (PROGRESS_STEP/10) == 0 && output_count > 0) begin
                        $display("INFO: MaxPool2D processing progress: %0d%% (%0d/%0d) at time %0t", 
                                (output_count * 100) / TOTAL_OUTPUT_FEATURES, 
                                output_count, TOTAL_OUTPUT_FEATURES, $time);
                    end
                    
                    // Update coordinates
                    if (out_col < OUT_WIDTH - 1) begin
                        out_col <= out_col + 1;
                    end else begin
                        out_col <= 0;
                        if (out_row < OUT_HEIGHT - 1) begin
                            out_row <= out_row + 1;
                        end else begin
                            out_row <= 0;
                            if (channel_idx < CHANNELS - 1) begin
                                channel_idx <= channel_idx + 1;
                            end else begin
                                // All processing complete
                                state <= DONE;
                                $display("DEBUG: MaxPool2D processing complete at time %0t", $time);
                            end
                        end
                    end
                end else begin
                    feature_valid_out <= 0;
                end
            end
            
            DONE: begin
                feature_valid_out <= 0;
                maxpool_done <= 1;
                $display("SUCCESS: MaxPool2D completed!");
                $display("Processed %0d input features to %0d output features", 
                        TOTAL_INPUT_FEATURES, TOTAL_OUTPUT_FEATURES);
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
        $display("DEBUG: MaxPool2D State change to %s at time %0t", 
                 (state == IDLE) ? "IDLE" :
                 (state == LOAD_DATA) ? "LOAD_DATA" :
                 (state == PROCESS) ? "PROCESS" :
                 (state == DONE) ? "DONE" : "UNKNOWN", $time);
        prev_state <= state;
    end
end

endmodule
