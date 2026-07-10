#import <Metal/Metal.h>
#import "metal_backend.h"
#import <stdio.h>
#import <string.h>

static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLComputePipelineState> forwardPipeline = nil;

bool init_metal_engine(void) {
    // Grab the system's default Metal GPU (e.g. M1/M2/M3 Max)
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
        printf("[METAL] FATAL: Metal is not supported on this device.\n");
        return false;
    }
    
    commandQueue = [device newCommandQueue];
    
    // Dynamically compile the shader library at runtime
    NSError *error = nil;
    NSString *shaderPath = @"shaders.metal";
    NSString *shaderSource = [NSString stringWithContentsOfFile:shaderPath encoding:NSUTF8StringEncoding error:&error];
    
    if (!shaderSource) {
        printf("[METAL] ERROR: Failed to load shaders.metal. Ensure it is in the working directory.\n");
        return false;
    }
    
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
    if (!library) {
        printf("[METAL] ERROR: Failed to compile Metal shaders: %s\n", [[error localizedDescription] UTF8String]);
        return false;
    }
    
    // Bind the specific GPU Kernel function
    id<MTLFunction> forwardFunction = [library newFunctionWithName:@"forward_pass_layer"];
    forwardPipeline = [device newComputePipelineStateWithFunction:forwardFunction error:&error];
    if (!forwardPipeline) {
        printf("[METAL] ERROR: Failed to create compute pipeline state.\n");
        return false;
    }
    
    printf("[METAL] SUCCESS: GPU Compute Engine Initialized! Bound to: %s\n", [[device name] UTF8String]);
    return true;
}

const char* get_metal_device_name(void) {
    if (device) return [[device name] UTF8String];
    return "Unknown Device";
}

// Submits a matrix multiplication task to the GPU queue
void metal_forward_layer(const float* inputs, 
                         const float* weights, 
                         const float* biases, 
                         float* outputs, 
                         int input_size, 
                         int output_size,
                         bool use_sigmoid) 
{
    @autoreleasepool {
        if (!device) return;
        
        // 1. Create Buffers (Push memory from CPU RAM to GPU VRAM)
        id<MTLBuffer> inBuffer = [device newBufferWithBytes:inputs length:input_size * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> wBuffer = [device newBufferWithBytes:weights length:input_size * output_size * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuffer = [device newBufferWithBytes:biases length:output_size * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuffer = [device newBufferWithLength:output_size * sizeof(float) options:MTLResourceStorageModeShared];
        
        int is_sig = use_sigmoid ? 1 : 0;
        
        // 2. Setup GPU Command Encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:forwardPipeline];
        [encoder setBuffer:inBuffer offset:0 atIndex:0];
        [encoder setBuffer:wBuffer offset:0 atIndex:1];
        [encoder setBuffer:bBuffer offset:0 atIndex:2];
        [encoder setBuffer:outBuffer offset:0 atIndex:3];
        [encoder setBytes:&input_size length:sizeof(int) atIndex:4];
        [encoder setBytes:&output_size length:sizeof(int) atIndex:5];
        [encoder setBytes:&is_sig length:sizeof(int) atIndex:6];
        
        // 3. Dispatch GPU Threads (1 Thread per Output Neuron, fully parallel)
        MTLSize gridSize = MTLSizeMake(output_size, 1, 1);
        NSUInteger threadGroupSize = forwardPipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > output_size) threadGroupSize = output_size;
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        // 4. Execute on GPU and Wait for Completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // 5. Pull results back from GPU VRAM to CPU RAM
        memcpy(outputs, [outBuffer contents], output_size * sizeof(float));
    }
}
