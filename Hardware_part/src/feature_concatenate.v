`timescale 1ns/1ps

module feature_concatenate #(
    parameter HEIGHT = 32,           // Feature map height
    parameter WIDTH = 32,            // Feature map width
    parameter IN1_CHANNELS = 512,    // First input channels (from Conv2DTranspose)
    parameter IN2_CHANNELS = 512,    // Second input channels (from skip connection)
    parameter DATA_WIDTH = 16        // Data width for each feature
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    
    // First input stream (from Conv2DTranspose)
    input wire [DATA_WIDTH-1:0] feature_in1,
    input wire feature_valid1,
    
    // Second input stream (from skip connection)
    input wire [DATA_WIDTH-1:0] feature_in2,
    input wire feature_valid2,
    
    // Output stream
    output reg [DATA_WIDTH-1:0] feature_out,
    output reg feature_valid_out,
    output reg concat_done
);

// Calculate total features
localparam OUT_CHANNELS = IN1_CHANNELS + IN2_CHANNELS;  // 512 + 512 = 1024
localparam TOTAL_IN1_FEATURES = HEIGHT * WIDTH * IN1_CHANNELS;   // 32*32*512 = 524,288
localparam TOTAL_IN2_FEATURES = HEIGHT * WIDTH * IN2_CHANNELS;   // 32*32*512 = 524,288
localparam TOTAL_OUT_FEATURES = HEIGHT * WIDTH * OUT_CHANNELS;   // 32*32*1024 = 1,048,576

// Memory for storing input features
reg [DATA_WIDTH-1:0] input1_buffer [0:TOTAL_IN1_FEATURES-1];
reg [DATA_WIDTH-1:0] input2_buffer [0:TOTAL_IN2_FEATURES-1];

// State machine
reg [2:0] state;
localparam IDLE = 3'b000;
localparam LOAD_INPUT1 = 3'b001;
localparam LOAD_INPUT2 = 3'b010;
localparam CONCATENATE = 3'b011;
localparam DONE = 3'b100;

// Counters
reg [31:0] input1_count;
reg [31:0] input2_count;
reg [31:0] output_count;

// Concatenation coordinates
reg [31:0] current_row, current_col;
reg [31:0] current_channel;

// Progress tracking
reg [31:0] progress_counter;
localparam PROGRESS_STEP = TOTAL_IN1_FEATURES / 100;

// Helper function to get spatial index (row, col)
function [31:0] get_spatial_index;
    input [31:0] row, col;
    begin
        get_spatial_index = (row * WIDTH) + col;
    end
endfunction

// Helper function to get buffer index for input1
function [31:0] get_input1_index;
    input [31:0] row, col, channel;
    begin
        get_input1_index = (channel * HEIGHT * WIDTH) + (row * WIDTH) + col;
    end
endfunction

// Helper function to get buffer index for input2
function [31:0] get_input2_index;
    input [31:0] row, col, channel;
    begin
        get_input2_index = (channel * HEIGHT * WIDTH) + (row * WIDTH) + col;
    end
endfunction

// String for state display
reg [8*12:1] state_name;
always @(*) begin
    case (state)
        IDLE: state_name = "IDLE";
        LOAD_INPUT1: state_name = "LOAD_INPUT1";
        LOAD_INPUT2: state_name = "LOAD_INPUT2";
        CONCATENATE: state_name = "CONCATENATE";
        DONE: state_name = "DONE";
        default: state_name = "UNKNOWN";
    endcase
end

// Main FSM
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        input1_count <= 0;
        input2_count <= 0;
        output_count <= 0;
        current_row <= 0;
        current_col <= 0;
        current_channel <= 0;
        feature_out <= 0;
        feature_valid_out <= 0;
        concat_done <= 0;
        progress_counter <= 0;
        
        $display("DEBUG: Feature Concatenation Reset");
        $display("Input1 size: %0dx%0dx%0d (%0d features)", HEIGHT, WIDTH, IN1_CHANNELS, TOTAL_IN1_FEATURES);
        $display("Input2 size: %0dx%0dx%0d (%0d features)", HEIGHT, WIDTH, IN2_CHANNELS, TOTAL_IN2_FEATURES);
        $display("Output size: %0dx%0dx%0d (%0d features)", HEIGHT, WIDTH, OUT_CHANNELS, TOTAL_OUT_FEATURES);
        
    end else begin
        case (state)
            IDLE: begin
                concat_done <= 0;
                feature_valid_out <= 0;
                if (start) begin
                    state <= LOAD_INPUT1;
                    input1_count <= 0;
                    input2_count <= 0;
                    output_count <= 0;
                    progress_counter <= 0;
                    $display("DEBUG: Starting feature concatenation at time %0t", $time);
                    $display("INFO: Expecting %0d features from input1 and %0d from input2", 
                            TOTAL_IN1_FEATURES, TOTAL_IN2_FEATURES);
                end
            end
            
            LOAD_INPUT1: begin
                if (feature_valid1) begin
                    input1_buffer[input1_count] <= feature_in1;
                    
                    // Progress reporting
                    if (input1_count % PROGRESS_STEP == 0 && input1_count > 0) begin
                        $display("INFO: Input1 loading progress: %0d%% (%0d/%0d) at time %0t", 
                                (input1_count * 100) / TOTAL_IN1_FEATURES, 
                                input1_count, TOTAL_IN1_FEATURES, $time);
                    end
                    
                    input1_count <= input1_count + 1;
                    
                    if (input1_count == TOTAL_IN1_FEATURES - 1) begin
                        state <= LOAD_INPUT2;
                        $display("DEBUG: Input1 loading complete, switching to Input2 at time %0t", $time);
                        $display("INFO: Loaded %0d features from input1", TOTAL_IN1_FEATURES);
                    end
                end
            end
            
            LOAD_INPUT2: begin
                if (feature_valid2) begin
                    input2_buffer[input2_count] <= feature_in2;
                    
                    // Progress reporting
                    if (input2_count % PROGRESS_STEP == 0 && input2_count > 0) begin
                        $display("INFO: Input2 loading progress: %0d%% (%0d/%0d) at time %0t", 
                                (input2_count * 100) / TOTAL_IN2_FEATURES, 
                                input2_count, TOTAL_IN2_FEATURES, $time);
                    end
                    
                    input2_count <= input2_count + 1;
                    
                    if (input2_count == TOTAL_IN2_FEATURES - 1) begin
                        state <= CONCATENATE;
                        current_row <= 0;
                        current_col <= 0;
                        current_channel <= 0;
                        output_count <= 0;
                        $display("DEBUG: Input2 loading complete, starting concatenation at time %0t", $time);
                        $display("INFO: Loaded %0d features from input2", TOTAL_IN2_FEATURES);
                    end
                end
            end
            
            CONCATENATE: begin
                // Output concatenated features in channel-first order for each spatial location
                if (output_count < TOTAL_OUT_FEATURES) begin
                    
                    if (current_channel < IN1_CHANNELS) begin
                        // Output from input1 buffer
                        feature_out <= input1_buffer[get_input1_index(current_row, current_col, current_channel)];
                        feature_valid_out <= 1;
                        
                    end else begin
                        // Output from input2 buffer (channels IN1_CHANNELS to OUT_CHANNELS-1)
                        feature_out <= input2_buffer[get_input2_index(current_row, current_col, current_channel - IN1_CHANNELS)];
                        feature_valid_out <= 1;
                    end
                    
                    output_count <= output_count + 1;
                    
                    // Progress reporting
                    if (output_count % (PROGRESS_STEP*2) == 0 && output_count > 0) begin
                        $display("INFO: Concatenation progress: %0d%% (%0d/%0d) at time %0t", 
                                (output_count * 100) / TOTAL_OUT_FEATURES, 
                                output_count, TOTAL_OUT_FEATURES, $time);
                    end
                    
                    // Update coordinates
                   if (current_channel < OUT_CHANNELS - 1) begin
    current_channel <= current_channel + 1;
end else begin
    current_channel <= 0;
    if (current_col < WIDTH - 1) begin
        current_col <= current_col + 1;
    end else begin
        current_col <= 0;
        if (current_row < HEIGHT - 1) begin
            current_row <= current_row + 1;
        end else begin
            // All features processed
            // Delay transition to DONE by one more cycle to assert feature_valid_out
            feature_valid_out <= 1; // keep it high for this cycle
            state <= DONE;
            $display("DEBUG: Concatenation complete at time %0t", $time);
        end
    end
end

                    
                end else begin
                    feature_valid_out <= 0;
                end
            end
            
            DONE: begin
                feature_valid_out <= 0;
                concat_done <= 1;
                $display("SUCCESS: Feature concatenation completed!");
                $display("Input1: %0dx%0dx%0d -> Input2: %0dx%0dx%0d -> Output: %0dx%0dx%0d", 
                        HEIGHT, WIDTH, IN1_CHANNELS, HEIGHT, WIDTH, IN2_CHANNELS, HEIGHT, WIDTH, OUT_CHANNELS);
                $display("Total features processed: %0d -> %0d", 
                        TOTAL_IN1_FEATURES + TOTAL_IN2_FEATURES, TOTAL_OUT_FEATURES);
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
        $display("DEBUG: Concatenation State change to %s at time %0t", state_name, $time);
        prev_state <= state;
    end
end

// Statistics tracking
reg [31:0] total_input1_received = 0;
reg [31:0] total_input2_received = 0;
reg [31:0] total_output_sent = 0;

always @(posedge clk) begin
    if (feature_valid1) total_input1_received <= total_input1_received + 1;
    if (feature_valid2) total_input2_received <= total_input2_received + 1;
    if (feature_valid_out) total_output_sent <= total_output_sent + 1;
end

// Final verification
always @(posedge concat_done) begin
    $display("=== Final Concatenation Statistics ===");
    $display("Input1 features received: %0d (expected: %0d)", total_input1_received, TOTAL_IN1_FEATURES);
    $display("Input2 features received: %0d (expected: %0d)", total_input2_received, TOTAL_IN2_FEATURES);
    $display("Output features sent: %0d (expected: %0d)", total_output_sent, TOTAL_OUT_FEATURES);
    
    if (total_input1_received == TOTAL_IN1_FEATURES && 
        total_input2_received == TOTAL_IN2_FEATURES && 
        total_output_sent == TOTAL_OUT_FEATURES) begin
        $display("✓ All feature counts verified!");
    end else begin
        $display("✗ Feature count mismatch detected!");
    end
end

endmodule
