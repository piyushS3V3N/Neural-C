# Neural-C: Complete Code & Library Documentation

This document explains the complete codebase of the Neural-C project, detailing each library used, the reasoning behind its inclusion, and the specific OS limitations.

## 1. Operating System Compatibility
**Supported OS:** macOS ONLY.

**Why?**
The core neural network includes an optimization layer written in Objective-C and Metal (`metal_backend.m`, `shaders.metal`). Metal is Apple's proprietary low-level, low-overhead hardware-accelerated 3D graphic and compute shader API. Since this codebase directly links against the macOS Metal framework to achieve GPU acceleration for the neural network's forward propagation, it fundamentally restricts the program to macOS environments.

## 2. Core Code Components

### `NeuralNetwork.c` & `NeuralNetwork.h`
- **Purpose**: Defines the `NeuralNetwork` struct and implements math logic. 
- **Features**: Includes Batched CPU algorithms (`forward_propagation_batch`, `backward_propagation_batch`) and delegates Apple Metal GEMM dispatch logic for lightning-fast tensor operations.

### `metal_backend.m` & `shaders.metal`
- **Purpose**: The GPU tensor engine.
- **Features**: Converts the sequential math into General Matrix Multiply (GEMM) parallel compute shaders across thousands of cores, caching massive memory matrices to eliminate PCIe bottlenecks.

### `visualizer.c` & `visualizer.h`
- **Purpose**: Handles the graphical rendering of the network's state.
- **Features**: Includes the live Sandbox logic, dynamic graphing of the MSE loss, interactive keyboard input, and the **Auto-Verifying Correction Engine** which intelligently forces the network to verify its own predictions during live retraining.

### `main.c`
- **Purpose**: The entry point of the application.
- **Functionality**: Initializes the window, loads data from `dataset.csv`, sets up the Neural Network and the Metal backend, and runs the main application loop. It handles persistent memory (loading/saving weights to `brain.bin`), manages user inputs (e.g., typing text to test the network, and pressing '1' or '0' for live-training on misclassifications), and coordinates the training and rendering loops.

### `dataset.csv`
- **Purpose**: A comprehensive training dataset generated from the NLTK names corpus (~8000 names).

### `brain.bin`
- **Purpose**: Binary file storing the network's persistent memory (weights and biases), automatically saved after training or live-correction.

## 3. Libraries Used

### **Standard C Libraries (`<stdio.h>`, `<stdlib.h>`, `<math.h>`, `<string.h>`, `<ctype.h>`)**
- **Why**: Used for standard memory management (`malloc`, `free`), mathematical operations (like `exp`, `log` for activation functions and loss calculation), file I/O operations (reading `dataset.csv`), and basic string manipulations.
- **OS**: Cross-platform.

### **Apple Metal API (`<Metal/Metal.h>`)**
- **Why**: Used to interface with the Mac's GPU. By offloading massive parallel tasks (like the matrix multiplications in neural network layers) to the GPU via compute shaders, the application achieves significantly higher performance than sequential CPU processing.
- **OS**: macOS only.

### **Raylib (`raylib.h`)**
- **Why**: A simple and easy-to-use library to enjoy videogames programming, but here used as a lightweight 2D rendering framework. It is responsible for creating the application window, rendering fonts (like Monaco), drawing shapes (lines for weights, circles for neurons), and capturing keyboard inputs. It was chosen because it allows for rapid UI development in C without the massive overhead of more complex GUI toolkits.
- **OS**: While Raylib itself is cross-platform, within this specific project context, it is compiled and linked for macOS.
