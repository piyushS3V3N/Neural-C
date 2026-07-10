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

static NSMutableDictionary *bufferCache = nil;

static id<MTLBuffer> get_cached_buffer(const void* ptr, int length, bool copy_data) {
    if (!bufferCache) bufferCache = [[NSMutableDictionary alloc] init];
    NSValue *key = [NSValue valueWithPointer:ptr];
    id<MTLBuffer> buf = [bufferCache objectForKey:key];
    if (!buf || [buf length] < length) {
        buf = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
        [bufferCache setObject:buf forKey:key];
    }
    if (copy_data) {
        memcpy([buf contents], ptr, length);
    }
    return buf;
}

void metal_forward_layer(const float* inputs, 
                         const float* weights, 
                         const float* biases, 
                         float* outputs, 
                         int input_size, 
                         int output_size,
                         bool use_sigmoid,
                         int batch_size) 
{
    @autoreleasepool {
        if (!device) return;
        
        // 1. Get Cached Buffers (Eliminates massive allocation overhead)
        id<MTLBuffer> inBuffer = get_cached_buffer(inputs, batch_size * input_size * sizeof(float), true);
        id<MTLBuffer> wBuffer = get_cached_buffer(weights, input_size * output_size * sizeof(float), true);
        id<MTLBuffer> bBuffer = get_cached_buffer(biases, output_size * sizeof(float), true);
        id<MTLBuffer> outBuffer = get_cached_buffer(outputs, batch_size * output_size * sizeof(float), false);
        
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
        [encoder setBytes:&batch_size length:sizeof(int) atIndex:7];
        
        // 3. Dispatch GPU Threads (1 Thread per Output Neuron, fully parallel)
        MTLSize gridSize = MTLSizeMake(batch_size, output_size, 1);
        NSUInteger w = forwardPipeline.threadExecutionWidth;
        NSUInteger h = forwardPipeline.maxTotalThreadsPerThreadgroup / w;
        MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        // 4. Execute on GPU and Wait for Completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // 5. Pull results back from GPU VRAM to CPU RAM
        memcpy(outputs, [outBuffer contents], batch_size * output_size * sizeof(float));
    }
}
