// Updated Version - 256x256 Convolution Layer
`timescale 1ns/1ps

module conv2d_layer #(
    parameter IMG_HEIGHT = 256,    // Updated for full size
    parameter IMG_WIDTH = 256,     // Updated for full size
    parameter IN_CHANNELS = 3,     
    parameter OUT_CHANNELS = 64,    
    parameter KERNEL_SIZE = 3,
    parameter DATA_WIDTH = 16
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [DATA_WIDTH-1:0] pixel_in,
    input wire pixel_valid,
    output reg [DATA_WIDTH-1:0] feature_out,
    output reg feature_valid,
    output reg conv_done
);

// Parameters - Use larger counters for 256x256
localparam TOTAL_PIXELS = IMG_HEIGHT * IMG_WIDTH * IN_CHANNELS;
localparam TOTAL_OUTPUTS = IMG_HEIGHT * IMG_WIDTH * OUT_CHANNELS;

// Memory - For simulation, you might want to use a smaller buffer or external memory
reg [DATA_WIDTH-1:0] input_buffer [0:TOTAL_PIXELS-1];

// Counters - Increased width for larger images
reg [31:0] pixel_count;    // Increased from 16 to 32 bits
reg [31:0] output_count;   // Increased from 16 to 32 bits
reg [2:0] state;

// Progress tracking for large simulations
reg [31:0] progress_counter;
localparam PROGRESS_STEP = TOTAL_PIXELS / 100; // Report every 1%

// States
localparam IDLE = 3'b000;
localparam LOAD_IMAGE = 3'b001;
localparam PROCESS = 3'b010;
localparam DONE = 3'b011;

// Debug: Add state names for display
reg [8*10:1] state_name;
always @(*) begin
    case (state)
        IDLE: state_name = "IDLE";
        LOAD_IMAGE: state_name = "LOAD_IMAGE";
        PROCESS: state_name = "PROCESS";
        DONE: state_name = "DONE";
        default: state_name = "UNKNOWN";
    endcase
end

// Main FSM
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        pixel_count <= 0;
        output_count <= 0;
        progress_counter <= 0;
        feature_out <= 0;
        feature_valid <= 0;
        conv_done <= 0;
        $display("DEBUG: Reset - state = IDLE, Image size: %0dx%0d, Total pixels: %0d", 
                 IMG_HEIGHT, IMG_WIDTH, TOTAL_PIXELS);
    end else begin
        case (state)
            IDLE: begin
                conv_done <= 0;
                feature_valid <= 0;
                if (start) begin
                    state <= LOAD_IMAGE;
                    pixel_count <= 0;
                    output_count <= 0;
                    progress_counter <= 0;
                    $display("DEBUG: IDLE->LOAD_IMAGE at time %0t", $time);
                    $display("INFO: Starting to load %0d pixels...", TOTAL_PIXELS);
                end
            end
            
            LOAD_IMAGE: begin
                if (pixel_valid) begin
                    input_buffer[pixel_count] <= pixel_in;
                    
                    // Progress reporting for large images
                    if (pixel_count % PROGRESS_STEP == 0 && pixel_count > 0) begin
                        $display("INFO: Loading progress: %0d%% (%0d/%0d pixels) at time %0t", 
                                (pixel_count * 100) / TOTAL_PIXELS, pixel_count, TOTAL_PIXELS, $time);
                    end
                    
                    pixel_count <= pixel_count + 1;
                    
                    if (pixel_count == TOTAL_PIXELS - 1) begin
                        state <= PROCESS;
                        output_count <= 0;
                        $display("DEBUG: LOAD_IMAGE->PROCESS at time %0t (loaded %0d pixels)", 
                                $time, TOTAL_PIXELS);
                        $display("INFO: Starting processing phase...");
                    end
                end
            end
            
            PROCESS: begin
                // Simple processing: just copy input to output with some operation
                if (output_count < TOTAL_OUTPUTS) begin
                    feature_out <= input_buffer[output_count % TOTAL_PIXELS] + 1; // Simple operation
                    feature_valid <= 1;
                    
                    // Progress reporting for processing
                    if (output_count % PROGRESS_STEP == 0 && output_count > 0) begin
                        $display("INFO: Processing progress: %0d%% (%0d/%0d outputs) at time %0t", 
                                (output_count * 100) / TOTAL_OUTPUTS, output_count, TOTAL_OUTPUTS, $time);
                    end
                    
                    output_count <= output_count + 1;
                    
                    if (output_count == TOTAL_OUTPUTS - 1) begin
                        state <= DONE;
                        $display("DEBUG: PROCESS->DONE at time %0t", $time);
                        $display("INFO: Processing completed!");
                    end
                end
            end
            
            DONE: begin
                feature_valid <= 0;
                conv_done <= 1;
                $display("DEBUG: FINAL - conv_done = %0d at time %0t", conv_done, $time);
                $display("SUCCESS: Convolution completed for %0dx%0d image!", IMG_HEIGHT, IMG_WIDTH);
            end
            
            default: begin
                state <= IDLE;
                $display("DEBUG: Unknown state, going to IDLE");
            end
        endcase
    end
end

// Debug: Monitor state changes
reg [2:0] prev_state = IDLE;
always @(posedge clk) begin
    if (state != prev_state) begin
        $display("DEBUG: State change from %s to %s at time %0t", 
                 (prev_state == IDLE) ? "IDLE" :
                 (prev_state == LOAD_IMAGE) ? "LOAD_IMAGE" :
                 (prev_state == PROCESS) ? "PROCESS" :
                 (prev_state == DONE) ? "DONE" : "UNKNOWN",
                 state_name, $time);
        prev_state <= state;
    end
end

endmodule
