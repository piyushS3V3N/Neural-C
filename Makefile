CC = gcc
CFLAGS = -O3 -Iinclude -Wall
FRAMEWORKS = -framework Metal -framework Foundation
RAYLIB_FLAGS = -I/opt/homebrew/Cellar/raylib/6.0/include -L/opt/homebrew/Cellar/raylib/6.0/lib -lraylib -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo

SRCS_LLM = src/tokenizer.c src/tensor.c src/gguf_parser.c src/math_kernels.c src/llm_metal_backend.m src/transformer_engine.c src/llm_trainer.c
SRCS_VISUALIZER = src/NeuralNetwork.c src/LLMClassifier.c src/visualizer.c src/metal_backend.m $(SRCS_LLM)

.PHONY: all llm trainer visualizer test clean

all: llm trainer visualizer

llm:
	$(CC) $(CFLAGS) apps/llm_main.c $(SRCS_LLM) $(FRAMEWORKS) -o neural_c_llm

trainer:
	$(CC) $(CFLAGS) apps/llm_trainer_main.c $(SRCS_LLM) $(FRAMEWORKS) -o neural_c_llm_trainer

visualizer:
	$(CC) $(CFLAGS) apps/main.c $(SRCS_VISUALIZER) $(RAYLIB_FLAGS) $(FRAMEWORKS) -o nn_visualizer

test:
	@echo "========================================="
	@echo "  Running Neural-C LLM Engine Test Suite "
	@echo "========================================="
	$(CC) $(CFLAGS) tests/test_tokenizer.c src/tokenizer.c -o test_tok && ./test_tok && rm -f test_tok
	$(CC) $(CFLAGS) tests/test_module2.c src/tensor.c -o test_m2 && ./test_m2 && rm -f test_m2
	$(CC) $(CFLAGS) tests/test_module3.c src/gguf_parser.c src/tokenizer.c -o test_m3 && ./test_m3 && rm -f test_m3
	$(CC) $(CFLAGS) tests/test_module4.c src/math_kernels.c src/tensor.c -o test_m4 && ./test_m4 && rm -f test_m4
	$(CC) $(CFLAGS) tests/test_module5.c src/math_kernels.c src/tensor.c src/llm_metal_backend.m $(FRAMEWORKS) -o test_m5 && ./test_m5 && rm -f test_m5
	$(CC) $(CFLAGS) tests/test_module6.c $(SRCS_LLM) $(FRAMEWORKS) -o test_m6 && ./test_m6 && rm -f test_m6
	$(CC) $(CFLAGS) tests/test_module7.c $(SRCS_LLM) $(FRAMEWORKS) -o test_m7 && ./test_m7 && rm -f test_m7
	@echo "\n>>> ALL 7 MODULE TESTS PASSED SUCCESSFULLY! <<<\n"

clean:
	rm -f neural_c_llm neural_c_llm_trainer nn_visualizer test_tok test_m2 test_m3 test_m4 test_m5 test_m6 test_m7 custom_llm_model.bin
