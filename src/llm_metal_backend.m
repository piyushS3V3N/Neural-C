#import <Metal/Metal.h>
#import "llm_metal_backend.h"
#import <stdio.h>
#import <string.h>

static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLComputePipelineState> gemvQ4Pipeline = nil;
static id<MTLComputePipelineState> gemvQ8Pipeline = nil;
static id<MTLComputePipelineState> gemvFP32Pipeline = nil;
static NSMutableDictionary *bufferCache = nil;

bool init_llm_metal_engine(void) {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
        printf("[METAL LLM] FATAL: Metal GPU default device not found.\n");
        return false;
    }
    
    commandQueue = [device newCommandQueue];
    bufferCache = [[NSMutableDictionary alloc] init];

    NSError *error = nil;
    NSString *shaderSource = [NSString stringWithContentsOfFile:@"shaders/llm_shaders.metal" encoding:NSUTF8StringEncoding error:&error];
    if (!shaderSource) {
        shaderSource = [NSString stringWithContentsOfFile:@"llm_shaders.metal" encoding:NSUTF8StringEncoding error:&error];
    }
    
    if (!shaderSource) {
        printf("[METAL LLM] ERROR: Failed to read llm_shaders.metal (checked shaders/llm_shaders.metal and ./llm_shaders.metal).\n");
        return false;
    }
    
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
    if (!library) {
        printf("[METAL LLM] ERROR: Failed to compile MSL shaders: %s\n", [[error localizedDescription] UTF8String]);
        return false;
    }
    
    id<MTLFunction> gemvQ4Func = [library newFunctionWithName:@"gemv_q4_0"];
    gemvQ4Pipeline = [device newComputePipelineStateWithFunction:gemvQ4Func error:&error];
    
    id<MTLFunction> gemvQ8Func = [library newFunctionWithName:@"gemv_q8_0"];
    gemvQ8Pipeline = [device newComputePipelineStateWithFunction:gemvQ8Func error:&error];
    
    id<MTLFunction> gemvFP32Func = [library newFunctionWithName:@"gemv_fp32"];
    gemvFP32Pipeline = [device newComputePipelineStateWithFunction:gemvFP32Func error:&error];
    
    if (!gemvQ4Pipeline || !gemvQ8Pipeline || !gemvFP32Pipeline) {
        printf("[METAL LLM] ERROR: Failed to initialize GPU compute pipelines.\n");
        return false;
    }
    
    printf("[METAL LLM] SUCCESS: GPU Compute Engine Initialized on: %s\n", [[device name] UTF8String]);
    return true;
}

const char* get_llm_metal_device_name(void) {
    if (device) return [[device name] UTF8String];
    return "Unknown Metal Device";
}

static id<MTLBuffer> get_cached_buffer(const void* ptr, size_t length, bool copy_data) {
    if (!ptr || length == 0) return nil;
    NSValue *key = [NSValue valueWithPointer:ptr];
    id<MTLBuffer> buf = [bufferCache objectForKey:key];
    if (!buf || [buf length] < length) {
        buf = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
        [bufferCache setObject:buf forKey:key];
        if (copy_data) {
            memcpy([buf contents], ptr, length);
        }
    }
    return buf;
}

void metal_gemv_q4_0(float* out, const float* x, const BlockQ4_0* w_q4, int in_dim, int out_dim) {
    if (!out) return;
    if (!w_q4 || !x || in_dim <= 0 || out_dim <= 0 || in_dim % 32 != 0) {
        if (out_dim > 0) memset(out, 0, out_dim * sizeof(float));
        return;
    }
    @autoreleasepool {
        if (!device) {
            matmul_q4_0(out, x, w_q4, in_dim, out_dim);
            return;
        }

        size_t blocks_per_row = in_dim / 32;
        size_t w_size_bytes = out_dim * blocks_per_row * sizeof(BlockQ4_0);
        
        id<MTLBuffer> xBuf = [device newBufferWithBytes:x length:in_dim * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> wBuf = get_cached_buffer(w_q4, w_size_bytes, true);
        id<MTLBuffer> outBuf = [device newBufferWithLength:out_dim * sizeof(float) options:MTLResourceStorageModeShared];
        
        if (!xBuf || !wBuf || !outBuf) {
            matmul_q4_0(out, x, w_q4, in_dim, out_dim);
            return;
        }

        id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:gemvQ4Pipeline];
        [encoder setBuffer:xBuf offset:0 atIndex:0];
        [encoder setBuffer:wBuf offset:0 atIndex:1];
        [encoder setBuffer:outBuf offset:0 atIndex:2];
        [encoder setBytes:&in_dim length:sizeof(int) atIndex:3];
        [encoder setBytes:&out_dim length:sizeof(int) atIndex:4];
        
        MTLSize gridSize = MTLSizeMake(out_dim, 1, 1);
        NSUInteger threadgroupSize = gemvQ4Pipeline.maxTotalThreadsPerThreadgroup;
        if (threadgroupSize > (NSUInteger)out_dim) threadgroupSize = out_dim;
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(threadgroupSize, 1, 1)];
        [encoder endEncoding];
        
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
        
        memcpy(out, [outBuf contents], out_dim * sizeof(float));
    }
}

void metal_gemv_q8_0(float* out, const float* x, const BlockQ8_0* w_q8, int in_dim, int out_dim) {
    if (!out) return;
    if (!w_q8 || !x || in_dim <= 0 || out_dim <= 0 || in_dim % 32 != 0) {
        if (out_dim > 0) memset(out, 0, out_dim * sizeof(float));
        return;
    }
    @autoreleasepool {
        if (!device) {
            matmul_q8_0(out, x, w_q8, in_dim, out_dim);
            return;
        }
        
        size_t blocks_per_row = in_dim / 32;
        size_t w_size_bytes = out_dim * blocks_per_row * sizeof(BlockQ8_0);
        
        id<MTLBuffer> xBuf = [device newBufferWithBytes:x length:in_dim * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> wBuf = get_cached_buffer(w_q8, w_size_bytes, true);
        id<MTLBuffer> outBuf = [device newBufferWithLength:out_dim * sizeof(float) options:MTLResourceStorageModeShared];
        
        if (!xBuf || !wBuf || !outBuf) {
            matmul_q8_0(out, x, w_q8, in_dim, out_dim);
            return;
        }

        id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:gemvQ8Pipeline];
        [encoder setBuffer:xBuf offset:0 atIndex:0];
        [encoder setBuffer:wBuf offset:0 atIndex:1];
        [encoder setBuffer:outBuf offset:0 atIndex:2];
        [encoder setBytes:&in_dim length:sizeof(int) atIndex:3];
        [encoder setBytes:&out_dim length:sizeof(int) atIndex:4];
        
        MTLSize gridSize = MTLSizeMake(out_dim, 1, 1);
        NSUInteger threadgroupSize = gemvQ8Pipeline.maxTotalThreadsPerThreadgroup;
        if (threadgroupSize > (NSUInteger)out_dim) threadgroupSize = out_dim;
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(threadgroupSize, 1, 1)];
        [encoder endEncoding];
        
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
        
        memcpy(out, [outBuf contents], out_dim * sizeof(float));
    }
}

void metal_gemv_fp32(float* out, const float* x, const float* w, int in_dim, int out_dim) {
    if (!out) return;
    if (!w || !x || in_dim <= 0 || out_dim <= 0) {
        memset(out, 0, out_dim * sizeof(float));
        return;
    }
    @autoreleasepool {
        if (!device) {
            matmul_fp32(out, x, w, in_dim, out_dim);
            return;
        }
        
        id<MTLBuffer> xBuf = [device newBufferWithBytes:x length:in_dim * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> wBuf = get_cached_buffer(w, in_dim * out_dim * sizeof(float), true);
        id<MTLBuffer> outBuf = [device newBufferWithLength:out_dim * sizeof(float) options:MTLResourceStorageModeShared];
        
        if (!xBuf || !wBuf || !outBuf) {
            matmul_fp32(out, x, w, in_dim, out_dim);
            return;
        }

        id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:gemvFP32Pipeline];
        [encoder setBuffer:xBuf offset:0 atIndex:0];
        [encoder setBuffer:wBuf offset:0 atIndex:1];
        [encoder setBuffer:outBuf offset:0 atIndex:2];
        [encoder setBytes:&in_dim length:sizeof(int) atIndex:3];
        [encoder setBytes:&out_dim length:sizeof(int) atIndex:4];
        
        MTLSize gridSize = MTLSizeMake(out_dim, 1, 1);
        NSUInteger threadgroupSize = gemvFP32Pipeline.maxTotalThreadsPerThreadgroup;
        if (threadgroupSize > (NSUInteger)out_dim) threadgroupSize = out_dim;
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(threadgroupSize, 1, 1)];
        [encoder endEncoding];
        
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
        
        memcpy(out, [outBuf contents], out_dim * sizeof(float));
    }
}
