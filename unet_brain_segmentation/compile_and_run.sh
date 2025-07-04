#!/bin/bash

echo "=== U-Net Brain Tumor Segmentation Simulation ==="
echo "Compiling Verilog files..."

# Compile all Verilog files
iverilog -o sim_results/unet_sim \
    -I src \
    src/unet_top.v \
    src/encoder_block.v \
    src/decoder_block.v \
    src/conv_block.v \
    src/conv2d_layer.v \
    src/conv2d_transpose.v \
    src/maxpool2d.v \
    src/batchnorm_relu.v \
    src/feature_concatenate.v \
    testbench/unet_top_tb.v

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Running simulation..."
    
    # Run simulation
    cd sim_results
    ./unet_sim
    
    echo "Simulation completed!"
    echo "Generated files:"
    ls -la *.vcd *.txt 2>/dev/null || echo "No output files generated"
    
    # Open waveform if GTKWave is available
    if command -v gtkwave &> /dev/null; then
        echo "Opening waveforms in GTKWave..."
        gtkwave unet_brain_segmentation.vcd &
    fi
else
    echo "Compilation failed! Check error messages above."
    exit 1
fi
