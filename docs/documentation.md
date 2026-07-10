# Neural-C: Complete Code & Library Documentation

This document explains the complete codebase of the Neural-C project, detailing each library used, the reasoning behind its inclusion, and the specific OS limitations.

## 1. Operating System Compatibility
**Supported OS:** macOS ONLY.

**Why?**
The core neural network includes an optimization layer written in Objective-C and Metal (`metal_backend.m`, `shaders.metal`). Metal is Apple's proprietary low-level, low-overhead hardware-accelerated 3D graphic and compute shader API. Since this codebase directly links against the macOS Metal framework to achieve GPU acceleration for the neural network's forward propagation, it fundamentally restricts the program to macOS environments.

## 2. Core Code Components

### `NeuralNetwork.c` / `NeuralNetwork.h`
- **Purpose**: Contains the pure C implementation of the Deep Neural Network (DNN).
- **Structure**: It defines a 5-layer MLP (Multi-Layer Perceptron) architecture (Inputs: 5, Hidden Layers: 16, 12, 8, 4, Output: 1).
- **Functionality**: Handles memory allocation, standard forward propagation (CPU), backward propagation (gradient descent), and binary cross-entropy (BCE) loss calculations.

### `metal_backend.m` / `metal_backend.h` / `shaders.metal`
- **Purpose**: The GPU acceleration backend.
- **Functionality**: `metal_backend.m` bridges the C code with Apple's Metal API. `shaders.metal` contains the actual compute shader code (written in MSL - Metal Shading Language) that runs on the GPU to parallelize matrix multiplications for forward propagation.

### `visualizer.c` / `visualizer.h`
- **Purpose**: Handles the graphical rendering of the network's state.
- **Functionality**: Draws the nodes, connections (colored by weight values), and the loss graph over time to provide a real-time visualization of the learning process.

### `main.c`
- **Purpose**: The entry point of the application.
- **Functionality**: Initializes the window, loads data from `dataset.csv`, sets up the Neural Network and the Metal backend, and runs the main application loop. It handles user inputs (e.g., typing text to test the network dynamically) and coordinates the training and rendering loops.

### `dataset.csv`
- **Purpose**: The dataset used to train the neural network.

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
