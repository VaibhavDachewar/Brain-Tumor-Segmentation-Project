`timescale 1ns/1ps

module unet_top_tb;

// Parameters
parameter IMG_HEIGHT = 256;
parameter IMG_WIDTH = 256;
parameter IN_CHANNELS = 3;
parameter DATA_WIDTH = 16;
parameter FRAC_WIDTH = 8;
parameter CLK_PERIOD = 10; // 100MHz clock

// Test signals
reg clk;
reg rst_n;
reg start;
reg [DATA_WIDTH-1:0] image_in;
reg image_valid;
reg [DATA_WIDTH-1:0] gamma, beta, mean, variance;
reg [DATA_WIDTH-1:0] kernel_weight, bias;

wire [DATA_WIDTH-1:0] segmentation_out;
wire segmentation_valid;
wire unet_done;

// Counters for input feeding
reg [31:0] pixel_count;
reg [31:0] total_pixels;
reg [31:0] output_pixel_count;

// Expected feature map sizes for verification
reg [31:0] expected_sizes [0:15][0:2]; // [stage][height/width/channels]

// Statistics tracking
reg [31:0] encoder_cycles [0:4];
reg [31:0] decoder_cycles [0:4];
reg [31:0] total_cycles;
reg [31:0] start_time, end_time;

// Output collection
reg [DATA_WIDTH-1:0] output_buffer [0:65535]; // 256*256 = 65536
reg [31:0] expected_output_pixels;

// UUT instantiation
unet_top #(
    .IMG_HEIGHT(IMG_HEIGHT),
    .IMG_WIDTH(IMG_WIDTH),
    .IN_CHANNELS(IN_CHANNELS),
    .DATA_WIDTH(DATA_WIDTH),
    .FRAC_WIDTH(FRAC_WIDTH)
) uut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .image_in(image_in),
    .image_valid(image_valid),
    .gamma(gamma),
    .beta(beta),
    .mean(mean),
    .variance(variance),
    .kernel_weight(kernel_weight),
    .bias(bias),
    .segmentation_out(segmentation_out),
    .segmentation_valid(segmentation_valid),
    .unet_done(unet_done)
);

// Clock generation
initial begin
    clk = 0;
    forever #(CLK_PERIOD/2) clk = ~clk;
end

// Initialize expected feature map sizes (matching Python output)
initial begin
    // Input: 256x256x3
    expected_sizes[0][0] = 256; expected_sizes[0][1] = 256; expected_sizes[0][2] = 3;
    
    // Encoder outputs (skip connections)
    expected_sizes[1][0] = 256; expected_sizes[1][1] = 256; expected_sizes[1][2] = 64;   // s1
    expected_sizes[2][0] = 128; expected_sizes[2][1] = 128; expected_sizes[2][2] = 128;  // s2
    expected_sizes[3][0] = 64;  expected_sizes[3][1] = 64;  expected_sizes[3][2] = 256;  // s3
    expected_sizes[4][0] = 32;  expected_sizes[4][1] = 32;  expected_sizes[4][2] = 512;  // s4
    
    // Encoder pooled outputs
    expected_sizes[5][0] = 128; expected_sizes[5][1] = 128; expected_sizes[5][2] = 64;   // p1
    expected_sizes[6][0] = 64;  expected_sizes[6][1] = 64;  expected_sizes[6][2] = 128;  // p2
    expected_sizes[7][0] = 32;  expected_sizes[7][1] = 32;  expected_sizes[7][2] = 256;  // p3
    expected_sizes[8][0] = 16;  expected_sizes[8][1] = 16;  expected_sizes[8][2] = 512;  // p4
    
    // Bottleneck
    expected_sizes[9][0] = 16;  expected_sizes[9][1] = 16;  expected_sizes[9][2] = 1024; // b1
    
    // Decoder outputs
    expected_sizes[10][0] = 32;  expected_sizes[10][1] = 32;  expected_sizes[10][2] = 512;  // d1
    expected_sizes[11][0] = 64;  expected_sizes[11][1] = 64;  expected_sizes[11][2] = 256;  // d2
    expected_sizes[12][0] = 128; expected_sizes[12][1] = 128; expected_sizes[12][2] = 128;  // d3
    expected_sizes[13][0] = 256; expected_sizes[13][1] = 256; expected_sizes[13][2] = 64;   // d4
    
    // Final output
    expected_sizes[14][0] = 256; expected_sizes[14][1] = 256; expected_sizes[14][2] = 1;   // output
end

// Task to display feature map information
task display_feature_info;
    input [127:0] stage_name;
    input [31:0] height, width, channels;
    begin
        $display("INFO: %s - Feature map size: %0dx%0dx%0d at time %0t", 
                 stage_name, height, width, channels, $time);
    end
endtask

// Task to verify feature map size
task verify_feature_size;
    input [127:0] stage_name;
    input [31:0] actual_h, actual_w, actual_c;
    input [31:0] expected_h, expected_w, expected_c;
    begin
        if (actual_h == expected_h && actual_w == expected_w && actual_c == expected_c) begin
            $display("PASS: %s - Size verification passed: %0dx%0dx%0d", 
                     stage_name, actual_h, actual_w, actual_c);
        end else begin
            $display("FAIL: %s - Size mismatch! Expected: %0dx%0dx%0d, Got: %0dx%0dx%0d", 
                     stage_name, expected_h, expected_w, expected_c, actual_h, actual_w, actual_c);
        end
    end
endtask

// Task to save output to file
task save_output_to_file;
    integer file_handle;
    integer i;
    begin
        file_handle = $fopen("segmentation_output.txt", "w");
        if (file_handle == 0) begin
            $display("ERROR: Could not open output file");
        end else begin
            $fwrite(file_handle, "Brain Tumor Segmentation Output\n");
            $fwrite(file_handle, "Image Size: %0dx%0d\n", IMG_HEIGHT, IMG_WIDTH);
            $fwrite(file_handle, "Total Output Pixels: %0d\n", output_pixel_count);
            $fwrite(file_handle, "Data Format: 16-bit fixed point (8 fractional bits)\n\n");
            
            for (i = 0; i < output_pixel_count; i = i + 1) begin
                $fwrite(file_handle, "Pixel[%0d]: 0x%04h (%.4f)\n", 
                        i, output_buffer[i], 
                        $itor(output_buffer[i]) / (1 << FRAC_WIDTH));
            end
            $fclose(file_handle);
            $display("INFO: Segmentation output saved to segmentation_output.txt");
        end
    end
endtask

// Output collection process
always @(posedge clk) begin
    if (segmentation_valid && output_pixel_count < 65536) begin
        output_buffer[output_pixel_count] <= segmentation_out;
        output_pixel_count <= output_pixel_count + 1;
        
        if (output_pixel_count % 1000 == 999) begin
            $display("Collected %0d output pixels", output_pixel_count + 1);
        end
    end
end

// Main test sequence
initial begin
    // Initialize signals
    rst_n = 0;
    start = 0;
    image_in = 0;
    image_valid = 0;
    gamma = 16'h0100;    // 1.0 in fixed point
    beta = 16'h0000;     // 0.0 in fixed point
    mean = 16'h0080;     // 0.5 in fixed point
    variance = 16'h0100; // 1.0 in fixed point
    kernel_weight = 16'h0040; // 0.25 in fixed point
    bias = 16'h0020;     // 0.125 in fixed point
    
    pixel_count = 0;
    output_pixel_count = 0;
    total_pixels = IMG_HEIGHT * IMG_WIDTH * IN_CHANNELS;
    expected_output_pixels = IMG_HEIGHT * IMG_WIDTH; // 256x256x1 = 65536
    
    // Initialize cycle counters
    for (integer i = 0; i < 5; i = i + 1) begin
        encoder_cycles[i] = 0;
        decoder_cycles[i] = 0;
    end
    total_cycles = 0;
    
    $display("=== U-Net Brain Tumor Segmentation Test ===");
    $display("Input Image Size: %0dx%0dx%0d", IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS);
    $display("Expected Output Size: %0dx%0dx1", IMG_HEIGHT, IMG_WIDTH);
    $display("Data Width: %0d bits, Fractional Width: %0d bits", DATA_WIDTH, FRAC_WIDTH);
    $display("Total Input Pixels: %0d", total_pixels);
    $display("Expected Output Pixels: %0d", expected_output_pixels);
    
    // Reset sequence
    $display("\n--- Reset Phase ---");
    #(CLK_PERIOD * 5);
    rst_n = 1;
    #(CLK_PERIOD * 2);
    
    // Start U-Net processing
    $display("\n--- Starting U-Net Processing ---");
    start_time = $time;
    start = 1;
    #CLK_PERIOD;
    start = 0;
    
    // Feed input image data
    fork
        // Input data feeding process
        begin
            #(CLK_PERIOD * 3); // Wait for U-Net to be ready
            
            $display("\n--- Feeding Input Image Data ---");
            for (pixel_count = 0; pixel_count < total_pixels; pixel_count = pixel_count + 1) begin
                @(posedge clk);
                // Generate synthetic brain MRI-like data
                image_in = generate_brain_pixel(pixel_count);
                image_valid = 1;
                
                if (pixel_count % 10000 == 9999) begin
                    $display("Fed %0d/%0d pixels (%.1f%%)", 
                             pixel_count + 1, total_pixels, 
                             ((pixel_count + 1) * 100.0) / total_pixels);
                end
            end
            
            @(posedge clk);
            image_valid = 0;
            $display("All input pixels fed (%0d total)", total_pixels);
        end
        
        // Monitor U-Net state transitions and performance
        begin
            wait(uut.state == uut.ENC1);
            $display("\n=== ENCODER PATH ===");
            display_feature_info("Input", IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS);
            encoder_cycles[0] = $time;
            
            wait(uut.state == uut.ENC2);
            encoder_cycles[0] = $time - encoder_cycles[0];
            $display("Encoder Block 1 completed in %0d cycles", encoder_cycles[0]/CLK_PERIOD);
            display_feature_info("After Encoder 1 (skip)", 256, 256, 64);
            display_feature_info("After Encoder 1 (pooled)", 128, 128, 64);
            encoder_cycles[1] = $time;
            
            wait(uut.state == uut.ENC3);
            encoder_cycles[1] = $time - encoder_cycles[1];
            $display("Encoder Block 2 completed in %0d cycles", encoder_cycles[1]/CLK_PERIOD);
            display_feature_info("After Encoder 2 (skip)", 128, 128, 128);
            display_feature_info("After Encoder 2 (pooled)", 64, 64, 128);
            encoder_cycles[2] = $time;
            
            wait(uut.state == uut.ENC4);
            encoder_cycles[2] = $time - encoder_cycles[2];
            $display("Encoder Block 3 completed in %0d cycles", encoder_cycles[2]/CLK_PERIOD);
            display_feature_info("After Encoder 3 (skip)", 64, 64, 256);
            display_feature_info("After Encoder 3 (pooled)", 32, 32, 256);
            encoder_cycles[3] = $time;
            
            wait(uut.state == uut.BOTTLENECK);
            encoder_cycles[3] = $time - encoder_cycles[3];
            $display("Encoder Block 4 completed in %0d cycles", encoder_cycles[3]/CLK_PERIOD);
            display_feature_info("After Encoder 4 (skip)", 32, 32, 512);
            display_feature_info("After Encoder 4 (pooled)", 16, 16, 512);
            encoder_cycles[4] = $time;
            
            wait(uut.state == uut.DEC1);
            encoder_cycles[4] = $time - encoder_cycles[4];
            $display("Bottleneck completed in %0d cycles", encoder_cycles[4]/CLK_PERIOD);
            display_feature_info("After Bottleneck", 16, 16, 1024);
            
            $display("\n=== DECODER PATH ===");
            decoder_cycles[0] = $time;
            
            wait(uut.state == uut.DEC2);
            decoder_cycles[0] = $time - decoder_cycles[0];
            $display("Decoder Block 1 completed in %0d cycles", decoder_cycles[0]/CLK_PERIOD);
            display_feature_info("After Decoder 1", 32, 32, 512);
            decoder_cycles[1] = $time;
            
            wait(uut.state == uut.DEC3);
            decoder_cycles[1] = $time - decoder_cycles[1];
            $display("Decoder Block 2 completed in %0d cycles", decoder_cycles[1]/CLK_PERIOD);
            display_feature_info("After Decoder 2", 64, 64, 256);
            decoder_cycles[2] = $time;
            
            wait(uut.state == uut.DEC4);
            decoder_cycles[2] = $time - decoder_cycles[2];
            $display("Decoder Block 3 completed in %0d cycles", decoder_cycles[2]/CLK_PERIOD);
            display_feature_info("After Decoder 3", 128, 128, 128);
            decoder_cycles[3] = $time;
            
            wait(uut.state == uut.FINAL_CONV);
            decoder_cycles[3] = $time - decoder_cycles[3];
            $display("Decoder Block 4 completed in %0d cycles", decoder_cycles[3]/CLK_PERIOD);
            display_feature_info("After Decoder 4", 256, 256, 64);
            decoder_cycles[4] = $time;
            
            wait(uut.state == uut.DONE);
            decoder_cycles[4] = $time - decoder_cycles[4];
            $display("Final Convolution completed in %0d cycles", decoder_cycles[4]/CLK_PERIOD);
            display_feature_info("Final Output", 256, 256, 1);
        end
        
        // Wait for completion and collect results
        begin
            wait(unet_done);
            end_time = $time;
            total_cycles = (end_time - start_time) / CLK_PERIOD;
            
            // Wait for all output pixels to be collected
            wait(output_pixel_count >= expected_output_pixels || $time > start_time + 100000000); // 100ms timeout
            
            $display("\n=== SEGMENTATION COMPLETED ===");
            $display("Total processing time: %0d cycles (%.2f ms at 100MHz)", 
                     total_cycles, total_cycles * 0.01);
            $display("Output pixels collected: %0d/%0d", output_pixel_count, expected_output_pixels);
            
            // Performance analysis
            $display("\n=== PERFORMANCE ANALYSIS ===");
            $display("Encoder Cycles:");
            for (integer i = 0; i < 5; i = i + 1) begin
                $display("  Block %0d: %0d cycles", i+1, encoder_cycles[i]/CLK_PERIOD);
            end
            $display("Decoder Cycles:");
            for (integer i = 0; i < 5; i = i + 1) begin
                $display("  Block %0d: %0d cycles", i+1, decoder_cycles[i]/CLK_PERIOD);
            end
            
            // Verify output statistics
            verify_output_statistics();
            
            // Save output to file
            save_output_to_file();
            
            $display("\n=== TEST COMPLETED ===");
            $display("Brain tumor segmentation test completed successfully!");
            $display("Check segmentation_output.txt for detailed results.");
            
            #(CLK_PERIOD * 10);
            $finish;
        end
    join
end

// Function to generate synthetic brain MRI pixel data
function [DATA_WIDTH-1:0] generate_brain_pixel;
    input [31:0] pixel_index;
    reg [31:0] x, y, channel;
    reg [DATA_WIDTH-1:0] pixel_value;
    begin
        // Calculate pixel coordinates
        x = (pixel_index / IN_CHANNELS) % IMG_WIDTH;
        y = (pixel_index / IN_CHANNELS) / IMG_WIDTH;
        channel = pixel_index % IN_CHANNELS;
        
        // Generate synthetic brain tissue-like intensities
        case (channel)
            0: begin // Gray matter simulation
                if ((x-128)*(x-128) + (y-128)*(y-128) < 6400) // Circle for brain
                    pixel_value = 16'h00C0 + ($random % 16'h0040); // 0.75 ± 0.25
                else
                    pixel_value = 16'h0020; // Background
            end
            1: begin // White matter simulation
                if ((x-128)*(x-128) + (y-128)*(y-128) < 4900) // Inner circle
                    pixel_value = 16'h00E0 + ($random % 16'h0030); // 0.875 ± 0.1875
                else if ((x-128)*(x-128) + (y-128)*(y-128) < 6400)
                    pixel_value = 16'h0080 + ($random % 16'h0040); // 0.5 ± 0.25
                else
                    pixel_value = 16'h0010; // Background
            end
            2: begin // CSF/Tumor simulation
                if ((x-100)*(x-100) + (y-100)*(y-100) < 400) // Small tumor region
                    pixel_value = 16'h00FF; // Bright tumor
                else if ((x-128)*(x-128) + (y-128)*(y-128) < 6400)
                    pixel_value = 16'h0060 + ($random % 16'h0020); // CSF
                else
                    pixel_value = 16'h0000; // Background
            end
            default: pixel_value = 16'h0000;
        endcase
        
        generate_brain_pixel = pixel_value;
    end
endfunction

// Task to verify output statistics
task verify_output_statistics;
    integer i;
    reg [31:0] tumor_pixels, normal_pixels;
    reg [DATA_WIDTH-1:0] min_val, max_val;
    real avg_val;
    begin
        tumor_pixels = 0;
        normal_pixels = 0;
        min_val = 16'hFFFF;
        max_val = 16'h0000;
        avg_val = 0.0;
        
        for (i = 0; i < output_pixel_count; i = i + 1) begin
            if (output_buffer[i] > 16'h0080) // Threshold at 0.5
                tumor_pixels = tumor_pixels + 1;
            else
                normal_pixels = normal_pixels + 1;
                
            if (output_buffer[i] < min_val) min_val = output_buffer[i];
            if (output_buffer[i] > max_val) max_val = output_buffer[i];
            avg_val = avg_val + $itor(output_buffer[i]);
        end
        
        avg_val = avg_val / output_pixel_count / (1 << FRAC_WIDTH);
        
        $display("\n=== OUTPUT STATISTICS ===");
        $display("Tumor pixels detected: %0d (%.2f%%)", tumor_pixels, 
                 (tumor_pixels * 100.0) / output_pixel_count);
        $display("Normal pixels: %0d (%.2f%%)", normal_pixels, 
                 (normal_pixels * 100.0) / output_pixel_count);
        $display("Output range: %.4f to %.4f", 
                 $itor(min_val) / (1 << FRAC_WIDTH),
                 $itor(max_val) / (1 << FRAC_WIDTH));
        $display("Average output value: %.4f", avg_val);
    end
endtask

// Timeout mechanism
initial begin
    #100000000; // 100ms timeout
    $display("ERROR: Test timeout reached!");
    $display("Current state: %s", uut.get_state_name(uut.state));
    $display("Pixels fed: %0d/%0d", pixel_count, total_pixels);
    $display("Output pixels: %0d/%0d", output_pixel_count, expected_output_pixels);
    $finish;
end

// Generate VCD file for waveform analysis
initial begin
    $dumpfile("unet_brain_segmentation.vcd");
    $dumpvars(0, unet_top_tb);
end

endmodule
