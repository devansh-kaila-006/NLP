# ML Runtimes

**Source**: https://aman.ai/primers/ai/ml-runtimes
**Category**: Miscellaneous
**Word Count**: 10940

---

Introduction
Architecture Overview of On-Device ML Runtimes
Common Architectural Layers
Architecture by Runtime
TensorRT
Core ML
MLX (Apple MLX)
ONNX Runtime
ExecuTorch
LidarTLM
llama.cpp
TensorFlow Lite / Serving
TensorRT Deep Dive
Overview
Architecture
Implementation Details
Pros and Cons
Example Workflow
Suitable Applications
Core ML Deep Dive
Overview
Architecture
Supported Model Types
Implementation Details
Pros and Cons
Example Code Snippet
MLX Deep Dive
Overview
Architecture
Implementation Details
Pros and Cons
Example Code Snippet
ONNX Runtime Deep Dive
Overview
Architecture
Implementation Details
Pros and Cons
Example Code Snippet (Python)
Use in Edge / On-Device Scenarios
ExecuTorch Deep Dive
Overview
Architecture
Implementation Details
Pros and Cons
Example Workflow
Suitable Applications
LidarTLM Deep Dive
Overview
Architecture
Implementation Details
Pros and Cons
Example Pseudocode Flow
Suitable Applications
llama.cpp
Deep Dive
Overview
Architecture
Implementation Details
Pros and Cons
Example CLI Inference
Suitable Applications
TensorFlow Lite / TensorFlow Serving Deep Dive
Overview
TensorFlow Lite Architecture
Implementation Details
TensorFlow Serving (Short Overview)
Pros and Cons
Example Inference (Python - TFLite)
Suitable Applications
Comparative Analysis
General Characteristics
Model Formats and Conversion
Execution Model and Hardware Support
Optimization, Size, and Constraints
Flexibility, Debugging, and Ecosystem
Comparative Summary and Guidance
Feature Comparison Table
Strengths by Runtime
Runtime Selection Guidance
Final Thoughts
Related: Serialization Formats Across Runtimes
Protocol Buffers (Protobuf)
FlatBuffer
GGUF (GPT-generated GGML Unified Format)
Bytecode Format (ExecuTorch)
Comparative Analysis
Model Execution Lifecycle Across ML Runtimes
General Workflow: From Model to Inference
Model Training
Model Conversion
Model Loading
Memory Allocation
Inference Execution
Postprocessing & Output
Lifecycle Optimization (Optional but Critical)
Runtime-Specific Execution Lifecycles
TensorRT Execution Lifecycle (NVIDIA GPUs)
Core ML Execution Lifecycle (Apple Platforms)
MLX Execution Lifecycle (Apple Silicon)
ONNX Runtime Execution Lifecycle
ExecuTorch Execution Lifecycle (MCU/Embedded Focus)
LidarTLM Execution Lifecycle (LiDAR-Focused Embedded Stacks)
llama.cpp
Execution Lifecycle (Quantized LLMs)
TensorFlow Lite Execution Lifecycle
Related: CPU Operator Libraries/Backends
Overview
FBGEMM
(by Meta,; Server CPUs)
XNNPACK
(by Google,; both Server and Mobile CPUs)
What to Choose When?
Design Notes
Comparative Analysis
Further Reading
Citation
Introduction
As AI becomes increasingly integral to modern software applications, deploying models directly on devices—such as smartphones, embedded systems, wearables, and edge computing nodes—has gained prominence. This approach, known as
on-device machine learning
, enables faster inference, improved privacy, offline capabilities, and lower latency compared to cloud-based alternatives.
Several runtimes/inference engines have been developed to facilitate the efficient execution of ML models on diverse hardware architectures. These runtimes vary significantly in terms of platform compatibility, supported model formats, execution optimizations, and hardware acceleration. This primer covers a detailed comparison of key ML runtimes that support on-device inference:
TensorRT
Core ML
MLX (Apple MLX)
ONNX Runtime
ExecuTorch
LidarTLM
llama.cpp
TensorFlow Lite / TensorFlow Serving
This primer includes both general-purpose and specialized runtimes, ranging from Core ML and TensorFlow Lite to transformer-specific tools like
llama.cpp
and GPU-optimized engines such as TensorRT.
Architecture Overview of On-Device ML Runtimes
On-device machine learning runtimes are engineered to execute pre-trained models efficiently within the constraints of mobile devices, embedded platforms, and personal computers. Despite the diversity of runtimes, they typically share core architectural components that manage model parsing, hardware abstraction, and execution flow.
This section outlines common architectural patterns and then provides architecture summaries for each runtime discussed in this primer.
Common Architectural Layers
Most on-device ML runtimes follow a layered architecture consisting of the following components:
Model Loader / Parser
: Responsible for reading serialized model files (e.g.,
.mlmodel
,
.tflite
,
.onnx
,
.pt
, etc.) and converting them into an internal representation suitable for execution.
Serialization Format:
Defines how models are stored on disk. Most runtimes use specialized formats (e.g., FlatBuffer in TFLite, Protobuf in TensorFlow/ONNX). Protobuf offers fast binary encoding and structured metadata representation, and is common in ONNX (
.onnx
) and TensorFlow (
.pb
) models.
Intermediate Representation (IR)
: Some runtimes convert models into an internal graph or IR that enables further optimization and abstraction from the original framework.
Kernel / Operator Library
: A collection of pre-implemented mathematical operations (e.g., convolution, matmul, ReLU) that form the backbone of computation. These may be hand-optimized for specific CPU, GPU, NPU, or DSP targets.
Execution Engine / Scheduler
: Coordinates the evaluation of the computational graph, manages dependencies, and dispatches workloads to the appropriate hardware accelerators.
Hardware Abstraction Layer (HAL)
: Encapsulates hardware-specific APIs and provides runtime support for leveraging specialized units like Apple’s ANE, Qualcomm’s Hexagon DSP, or CUDA cores on NVIDIA GPUs.
Architecture by Runtime
TensorRT
Model Format
:
.plan
(TensorRT Engine)
Execution Flow
:
Accepts models in ONNX, TensorFlow, or Caffe formats
Optimizes and compiles model into a serialized CUDA engine (
.plan
)
Engine executes directly via CUDA on supported NVIDIA GPUs
Hardware Support
: NVIDIA GPUs (desktop, embedded, server)
Backend Design
: Layer fusion, kernel autotuning,
int8
/
float16
quantization, Tensor Cores
Strengths
: Extreme inference speed on NVIDIA hardware, minimal latency, quantization support
Weaknesses
: GPU-only, requires CUDA, less flexible for model updates at runtime
Core ML
Model Format
:
.mlmodel
, optionally converted from other formats using
coremltools
Execution Flow
:
Model is compiled into a Core ML model package (
.mlmodelc
)
Uses internal execution graph
Runtime determines target hardware (CPU, GPU, or ANE) dynamically
Hardware Support
: CPU, GPU, Apple Neural Engine (ANE)
Backend Design
: Proprietary graph engine, no direct user-accessible IR
Strengths
: Seamless Apple integration, high-level API, automatic hardware optimization
Weaknesses
: Apple-platform only, opaque architecture, limited transparency for debugging
MLX (Apple MLX)
Model Format
: Python-based tensor operations with PyTorch-like syntax
Execution Flow
:
Eager mode and graph execution both supported
Uses Metal Performance Shaders and ANE backend where possible
Hardware Support
: Primarily Apple Silicon (M-series CPU, GPU, ANE)
Backend Design
: Dynamic execution engine; uses MLX backend API
Strengths
: Developer flexibility, research-oriented, direct tensor ops
Weaknesses
: Early-stage, Apple-only, smaller community, fewer pre-built models
ONNX Runtime
Model Format
:
.onnx
Execution Flow
:
Loads ONNX graph and converts to optimized IR
Graph optimization passes applied (e.g., constant folding, fusion)
Execution providers (EPs) handle hardware-specific execution
Hardware Support
: CPU, GPU (CUDA, ROCm), NNAPI, DirectML, ARM, OpenVINO
Backend Design
: Pluggable EP system, modular kernel dispatch
Strengths
: Cross-platform, flexible, highly optimized
Weaknesses
: Model conversion may be lossy or complex, mobile-specific tuning needed
ExecuTorch
Model Format
: PyTorch Lite models,
ptc
compiled bytecode
Execution Flow
:
TorchScript traced models compiled using Ahead-of-Time (AOT) compiler
Produces a minimal runtime with only needed ops
Bytecode is executed on microcontroller or mobile device
Hardware Support
: CPU, MCU, potentially DSP/NPU
Backend Design
: AOT compiler, custom micro runtime, graph executor
Strengths
: Lightweight, optimized for resource-constrained environments
Weaknesses
: Limited model format support, newer toolchain
LidarTLM
Model Format
: Custom or converted models for lidar data processing
Execution Flow
:
Ingests sparse point cloud or voxel data
Uses spatial and temporal inference pipelines
Hardware Support
: ARM CPUs, embedded GPU, or AI co-processors
Backend Design
: Spatially-aware computation graph; sensor-fusion modules
Strengths
: Specialized for lidar, supports sensor fusion
Weaknesses
: Niche use case, limited community and documentation
llama.cpp
Model Format
: Quantized LLM formats (GGUF, etc.)
Execution Flow
:
Loads quantized model into memory
Performs batched matmul-based transformer inference
Multi-threaded CPU execution with optional GPU offload (via OpenCL, Metal)
Hardware Support
: CPU, optionally GPU
Backend Design
: Minimalist tensor framework, custom linear algebra, no IR
Strengths
: Extremely portable, optimized for low-RAM devices, self-contained
Weaknesses
: Focused only on LLMs, lower-level interface
TensorFlow Lite / Serving
Model Format
:
.tflite
(Lite),
.pb
or SavedModel (Serving)
Execution Flow
:
TFLite: uses FlatBuffer model, loads and interprets ops
Serving: REST/gRPC server for remote model inference
Hardware Support
:
TFLite: CPU, GPU, EdgeTPU, NNAPI, Hexagon DSP
Serving: Primarily server-side; not for on-device use
Backend Design
:
TFLite: statically compiled interpreters with kernel registry
TFLite delegates for hardware acceleration
Strengths
: Broad compatibility, active ecosystem, stable
Weaknesses
: Delegate configuration can be tricky, Serving not suitable for offline use
TensorRT Deep Dive
TensorRT is NVIDIA’s high-performance, low-latency inference runtime for deep learning models. It is purpose-built for GPU-accelerated inference and heavily optimized for NVIDIA’s hardware, including desktop GPUs, Jetson embedded boards, and datacenter GPUs with Tensor Cores.
Overview
Developer Target
: Engineers deploying deep learning models on NVIDIA hardware
Use Cases
: Vision inference, robotics, autonomous vehicles, embedded AI with Jetson, high-throughput servers
Model Format
: ONNX, Caffe, TensorFlow (converted to
.plan
engine)
Conversion Tools
:
trtexec
, TensorRT Python/C++ APIs
Architecture
TensorRT transforms trained models into an optimized engine using multiple optimization passes:
Execution Flow
:
Model Import
: Loads model (typically ONNX) using TensorRT parser
Optimization
:
Layer fusion
Precision calibration (
float16
,
int8
)
Kernel selection and scheduling
Engine Building
:
Generates a
.plan
file (serialized CUDA engine)
This engine can be reused for fast deployment
Inference Execution
:
Input data fed through pre-allocated CUDA buffers
Execution is entirely GPU-bound using CUDA streams
Key Components
:
Builder
: Optimizes and generates runtime engine
Runtime
: Loads and executes serialized engine
Execution Context
: Holds all buffers and workspace
Calibrator
: Generates
int8
quantization scale factors using sample data
Implementation Details
Quantization Support
:
float32
,
float16
, and
int8
precision modes
int8
requires calibration dataset (representative samples)
Layer Fusion
:
Combines ops like conv + bias + activation into a single kernel
Reduces memory overhead and execution latency
Dynamic Shapes
:
Supports engines that accept varying input sizes with shape profiles
Deployment
:
Supports inference from Python or C++
Compatible with DeepStream SDK, TensorRT-LLM, and Jetson platforms
Pros and Cons
Pros
:
Best-in-class GPU inference performance
Optimized for Tensor Cores (Ampere, Hopper, etc.)
Rich tooling (e.g.,
trtexec
, calibration tools)
Integration with Jetson for embedded AI
Cons
:
Requires NVIDIA GPU and CUDA runtime
Not suitable for CPU or cross-platform apps
Build/optimization pipeline adds complexity
Engine regeneration needed if input shape or model changes significantly
Example Workflow
Model Conversion (ONNX \(\rightarrow\) Engine):
trtexec
--onnx
=
model.onnx
--saveEngine
=
model.plan
--
`
float16
`
C++ Inference:
nvinfer1
::
IRuntime
*
runtime
=
nvinfer1
::
createInferRuntime
(
logger
);
std
::
ifstream
engineFile
(
"model.plan"
,
std
::
ios
::
binary
);
nvinfer1
::
ICudaEngine
*
engine
=
runtime
->
deserializeCudaEngine
(...);
Python Inference:
import
tensorrt
as
trt
TRT_LOGGER
=
trt
.
Logger
()
with
open
(
"model.plan"
,
"rb"
)
as
f
:
engine
=
trt
.
Runtime
(
TRT_LOGGER
).
deserialize_cuda_engine
(
f
.
read
())
Suitable Applications
Real-time object detection on Jetson Nano/Xavier
Batch inference in ML inference servers
int8
-quantized NLP models for chatbots
High-throughput video analytics (via DeepStream)
TensorRT excels in performance-critical scenarios where latency, batch throughput, or GPU utilization is a bottleneck. It’s a specialized, production-grade runtime for teams fully committed to NVIDIA’s platform.
Core ML Deep Dive
Core ML is Apple’s on-device machine learning framework, designed to provide seamless model deployment and execution across the Apple ecosystem. It’s tailored for iOS, macOS, watchOS, and tvOS, offering tight integration with system-level APIs and hardware acceleration units like the Apple Neural Engine (ANE).
Overview
Developer Target
: iOS/macOS developers
Use Cases
: Image recognition, natural language processing, AR/VR, real-time gesture and object detection
Model Format
:
.mlmodel
(converted to
.mlmodelc
at compile time)
Conversion Tools
:
coremltools
, Apple Create ML, ONNX to Core ML converters
Architecture
Model Compiler
: Converts
.mlmodel
to
.mlmodelc
, a compiled model package optimized for fast execution. It includes a serialized computation graph, weights, metadata, and hardware hints.
Execution Pipeline
:
Model Load
: App loads the
.mlmodelc
file at runtime using the
MLModel
API.
Prediction API
: Developer calls
prediction(input:)
, which triggers the internal compute graph.
Backend Selection
: Core ML dynamically selects the best available backend (CPU, GPU, ANE) based on model ops and hardware.
Execution Engine
: Executes the optimized graph using Apple’s proprietary kernel implementations.
Output
: Returns structured model output (class label, bounding box, etc.) as Swift-native objects.
Key Components
:
MLModel Interface
: Main interaction point for inference
MLMultiArray
: N-dimensional tensor abstraction
MLFeatureValue / MLFeatureProvider
: Input-output containers
NeuralNetwork.proto
: Defines underlying graph schema for neural network layers
Supported Model Types
Neural Networks (CNNs, RNNs, Transformers)
Decision Trees and Ensembles (from XGBoost, scikit-learn)
Natural Language models (tokenizers, embeddings)
Audio signal processing
Custom models using Core ML’s custom layers
Implementation Details
Conversion Process
:
Models from PyTorch, TensorFlow, scikit-learn, or XGBoost are first converted to ONNX or a supported format
coremltools.convert()
maps ops to Core ML equivalents and produces
.mlmodel
Optional model quantization (e.g., 16-bit float) can be applied to reduce size
Hardware Utilization
:
Automatically uses ANE if available (iPhone 8 and later)
Fallback to Metal GPU or CPU if ANE doesn’t support all ops
Internal heuristics determine fallback patterns and op partitioning
Custom Layers
:
Developers can define
MLCustomModel
classes
Useful when Core ML lacks certain ops
Requires manual tensor handling and native Swift/Obj-C implementation
Pros and Cons
Pros:
Deep Apple integration (Vision, AVFoundation, ARKit, etc.)
Seamless use of hardware accelerators
High-level Swift API for rapid development
Secure and privacy-focused (no data leaves device)
Optimized runtime with minimal latency
Cons:
Apple-only ecosystem
Conversion limitations (unsupported ops in some models)
Limited visibility into runtime internals
Custom layer interface can be verbose and inflexible
Example Code Snippet
guard
let
model
=
try
?
MyImageClassifier
(
configuration
:
MLModelConfiguration
())
else
{
fatalError
(
"Model failed to load"
)
}
let
input
=
try
?
MLMultiArray
(
shape
:
[
1
,
3
,
224
,
224
],
dataType
:
.
float32
)
// Fill input array with pixel data
let
output
=
try
?
model
.
prediction
(
input
:
input
!
)
print
(
output
?
.
classLabel
??
"Prediction failed"
)
MLX Deep Dive
MLX (Machine Learning eXperimentation) is a relatively new Apple-developed machine learning framework built specifically for Apple Silicon. It is designed for flexibility, research, and experimentation, offering a PyTorch-like Python API with eager and compiled execution. Unlike Core ML, which targets app integration and production deployment, MLX is meant for model development, prototyping, and edge inference—while taking full advantage of Apple hardware like the M-series chips.
Put simply, MLX is particularly well-suited for developers focused on rapid iteration and fine-tuning of models on Apple devices. It’s promising for LLMs and vision transformers on MacBooks and other Apple Silicon-powered hardware.
Overview
Developer Target
: ML researchers and developers using Apple Silicon
Use Cases
: Research, fine-tuning models on-device, LLM inference, Apple-optimized ML pipelines
Model Format
: No proprietary serialized model format; models are expressed in Python source code using
mlx.nn
layers
Conversion Tools
: Emerging support for PyTorch model import via
mlx-trace
and ONNX conversion
Architecture
MLX is a minimal and composable tensor library that uses Apple’s Metal Performance Shaders (MPS) and optionally the Apple Neural Engine (ANE) for hardware acceleration.
Execution Modes
:
Eager Execution
: Immediate computation for prototyping/debugging
Compiled Graph
: Via
mlx.compile()
for performance-critical inference
Core Components
:
mlx.core
: Tensor definitions and low-level math operations
mlx.nn
: High-level neural network module abstraction (analogous to PyTorch’s
nn.Module
)
mlx.optimizers
: Gradient-based optimizers for training
mlx.transforms
: Preprocessing utilities (e.g., normalization, resizing)
Hardware Abstraction
:
Primarily targets the GPU via MPS
MLX compiler performs static analysis to optimize kernel dispatch and memory usage
ANE support is still evolving and model-dependent
Implementation Details
Tensor Memory Model
:
MLX tensors are immutable
Operations generate new tensors rather than mutating in-place
Enables functional purity and easier graph compilation
JIT Compilation
:
While code is typically run in Python, MLX allows functions to be decorated with
@mlx.compile
to trace and compile computation graphs
Reduces memory allocations and kernel overhead
Custom Modules
:
Developers can create custom layers by subclassing
mlx.nn.Module
Supports standard layers like
Linear
,
Conv2d
,
LayerNorm
, etc.
Interoperability
:
MLX includes tools to convert PyTorch models using tracing (WIP)
No built-in ONNX or TensorFlow Lite importer yet, though development is ongoing
Pros and Cons
Pros:
Highly optimized for Apple Silicon (especially M1/M2)
Lightweight and minimalist API with functional programming style
Supports training and inference on-device
Fast experimentation with eager mode and compilation toggle
Tensor API is intuitive for PyTorch users
Cons:
Only runs on macOS with Apple Silicon (no iOS, no Windows/Linux)
Ecosystem still maturing (e.g., fewer pre-trained models, limited documentation)
No official deployment format—source code is the model
Interop with other frameworks is under active development but not production-ready
Example Code Snippet
import
mlx.core
as
mx
import
mlx.nn
as
nn
class
SimpleMLP
(
nn
.
Module
):
def
__init__
(
self
):
super
().
__init__
()
self
.
linear1
=
nn
.
Linear
(
784
,
256
)
self
.
relu
=
nn
.
ReLU
()
self
.
linear2
=
nn
.
Linear
(
256
,
10
)
def
__call__
(
self
,
x
):
x
=
self
.
linear1
(
x
)
x
=
self
.
relu
(
x
)
return
self
.
linear2
(
x
)
model
=
SimpleMLP
()
input
=
mx
.
random
.
normal
((
1
,
784
))
output
=
model
(
input
)
print
(
"Prediction:"
,
output
)
For accelerated inference:
compiled_fn
=
mx
.
compile
(
model
)
output
=
compiled_fn
(
input
)
ONNX Runtime Deep Dive
ONNX Runtime (ORT) is a cross-platform, high-performance inference engine for deploying models in the Open Neural Network Exchange (ONNX) format. Maintained by Microsoft, it is widely adopted due to its flexibility, extensibility, and support for numerous hardware backends. ONNX itself is an open standard that enables interoperability between ML frameworks like PyTorch, TensorFlow, and scikit-learn.
Overview
Developer Target
: Application developers, MLOps teams, platform architects
Use Cases
: Cross-framework inference, model portability, production deployments (cloud + edge), hardware acceleration
Model Format
:
.onnx
(Open Neural Network Exchange format)
Conversion Tools
:
torch.onnx.export
,
tf2onnx
,
skl2onnx
, and many others
Architecture
ONNX Runtime is structured around a pluggable and modular execution engine, making it suitable for CPU, GPU, and specialized accelerators. It uses an intermediate computation graph optimized at load time, and delegates computation to “Execution Providers” (EPs).
Execution Flow
:
Model Load
: Parses the
.onnx
model file into an internal graph representation.
Graph Optimization
: Applies a set of graph rewrite passes—like constant folding, node fusion, and dead node elimination.
Execution Provider Selection
: Based on available hardware and EP priorities, operators are assigned to execution backends.
Execution
: ORT schedules and dispatches kernel calls for each partition of the graph.
Output Handling
: Results are returned in native types or via C/C++/Python APIs.
Key Components
:
Session
:
InferenceSession
is the main object for loading and running models.
Execution Providers (EPs)
: Modular backend plugins such as:
CPU (default)
CUDA (NVIDIA GPUs)
DirectML (Windows GPU)
OpenVINO (Intel accelerators)
NNAPI (Android)
CoreML (iOS/macOS)
TensorRT
QNN (Qualcomm AI Engine)
Graph Transformer
: Rewrites and optimizes the computation graph
Kernel Registry
: Maps ONNX ops to optimized implementations
Implementation Details
Model Format
:
ONNX models are stored in protobuf format
Static computation graph with explicit type and shape information
Supports operator versioning to ensure backward compatibility
Customization
:
Developers can register custom ops and execution providers
Optional use of external initializers and custom inference contexts
Execution Optimization
:
Graph transformation level can be controlled (basic, extended, all)
EPs can share execution (e.g., some layers on CPU, others on GPU)
Quantization and sparsity-aware execution supported via tools like
onnxruntime-tools
Mobile Support
:
ONNX Runtime Mobile: A statically linked, size-reduced runtime
Works with Android and iOS, using NNAPI, Core ML, or CPU fallback
Pros and Cons
Pros:
Framework agnostic and highly interoperable
Broad hardware support via modular execution providers
Strong community and industrial backing (Microsoft, AWS, NVIDIA, etc.)
Mobile support with optimized builds and quantized execution
Extensive language bindings (Python, C++, C#, Java)
Cons:
Debugging can be complex across EPs
Conversion process from other frameworks may require custom scripts
ONNX opset compatibility issues can arise across versions
Mobile optimization (size, latency) requires manual tuning
Example Code Snippet (Python)
import
onnxruntime
as
ort
import
numpy
as
np
# Load ONNX model
session
=
ort
.
InferenceSession
(
"resnet50.onnx"
)
# Prepare input
input_name
=
session
.
get_inputs
()[
0
].
name
input_data
=
np
.
random
.
rand
(
1
,
3
,
224
,
224
).
astype
(
np
.
float32
)
# Run inference
outputs
=
session
.
run
(
None
,
{
input_name
:
input_data
})
print
(
"Prediction shape:"
,
outputs
[
0
].
shape
)
Using CUDA Execution Provider
:
session
=
ort
.
InferenceSession
(
"resnet50.onnx"
,
providers
=
[
'CUDAExecutionProvider'
])
Use in Edge / On-Device Scenarios
ONNX Runtime Mobile is specifically designed for deployment on edge devices. Key features include:
Stripped-down build (~1–2 MB)
FlatBuffer format support in preview
Android NNAPI and iOS Core ML integration
Prebuilt minimal runtime packages for specific models
ONNX Runtime is best suited for applications where:
Portability across hardware is essential
Mixed execution (CPU + accelerator) is beneficial
The model pipeline involves multiple frameworks
ExecuTorch Deep Dive
ExecuTorch is a lightweight runtime and deployment framework built by Meta (Facebook) to run PyTorch models on constrained edge devices, including microcontrollers (MCUs), embedded systems, and mobile hardware. It is designed with the principles of minimalism, portability, and execution efficiency. Unlike full PyTorch runtimes, ExecuTorch leverages Ahead-of-Time (AOT) compilation and produces compact bytecode representations of models.
Overview
Developer Target
: Embedded ML engineers, mobile and edge system developers
Use Cases
: Sensor fusion, vision at the edge, voice command detection, ultra-low-power AI applications
Model Format
: Compiled TorchScript bytecode (
.ptc
)
Conversion Tools
: PyTorch \(\rightarrow\) TorchScript \(\rightarrow\) ExecuTorch via AOT pipeline
Architecture
ExecuTorch redefines the execution pipeline for PyTorch models in low-resource environments. Its architecture includes a static graph compiler, a runtime interpreter, and pluggable dispatch interfaces for targeting different hardware backends.
Execution Flow
:
Model Export
:
Model defined in PyTorch and traced/scripted via TorchScript.
ExecuTorch’s AOT compiler converts it into a compact bytecode format.
Runtime Embedding
:
The bytecode and necessary ops are compiled with the target runtime.
Optional op pruning removes unused operations.
Deployment
:
Model and runtime are flashed onto the device.
Inference is run via a lightweight VM interpreter.
Key Components
:
Bytecode Format
:
.ptc
files contain compiled operators and control flow
VM Runtime
: A minimal interpreter that reads and executes bytecode
Dispatcher
: Routes ops to backend implementations
Memory Arena
: Static memory model, optionally no dynamic allocation
Implementation Details
AOT Compiler
:
Converts scripted TorchScript models into bytecode and op kernels
Includes a model linker that statically binds required ops
Can target C/C++ or platform-specific formats (Zephyr, FreeRTOS)
Operator Handling
:
Customizable op kernels allow device-specific optimization
Optional kernel fusion via compiler passes for performance
Runtime Constraints
:
Code size: Can be <500 KB with aggressive pruning
No reliance on dynamic memory allocation (static buffer planning)
Designed for devices with as little as 256 KB RAM
Integration
:
Written in C++
Can integrate with sensor pipelines, real-time OS, or MCU firmware
Open-sourced with tooling for building and flashing models to hardware
Pros and Cons
Pros:
Extremely lightweight, MCU-ready
AOT compilation reduces runtime overhead
Deterministic memory usage (good for real-time applications)
Modular and open-source with low-level control
PyTorch-compatible workflow for training and export
Cons:
Requires model to be written in a static subset of PyTorch
Limited dynamic control flow (must be scriptable)
Debugging and tooling less mature than mainstream PyTorch or TensorFlow Lite
Focused on inference only; no training support on-device
Example Workflow
Model Export (Python):
import
torch
import
torch.nn
as
nn
class
TinyModel
(
nn
.
Module
):
def
__init__
(
self
):
super
().
__init__
()
self
.
fc
=
nn
.
Linear
(
4
,
2
)
def
forward
(
self
,
x
):
return
self
.
fc
(
x
)
model
=
TinyModel
()
scripted
=
torch
.
jit
.
script
(
model
)
scripted
.
save
(
"model.pt"
)
ExecuTorch AOT Compilation (CLI or CMake):
executorchc compile
--model
model.pt
--output
model.ptc
--target
cortex-m
Embedded Runtime Integration (C++):
#include
"executorch/runtime/runtime.h"
executorch
::
load_model
(
"model.ptc"
);
executorch
::
run_model
(
input_tensor
,
output_tensor
);
Suitable Applications
Wake-word detection on MCUs
Gesture recognition using MEMS sensors
Smart agriculture (tiny vision models)
Battery-powered health monitoring devices
ExecuTorch fills a critical niche for deploying PyTorch-trained models on hardware where traditional runtimes like TensorFlow Lite or ONNX Runtime are too heavy.
LidarTLM Deep Dive
LidarTLM (LiDAR Tensor Layer Module) is a specialized, lower-profile runtime or processing pipeline designed for inference on LiDAR data using neural networks. It is not a mainstream or widely standardized runtime like TensorFlow Lite or ONNX Runtime, but rather refers to a class of embedded software tools tailored for 3D point cloud inference and fusion with temporal data—typically in autonomous systems, robotics, or advanced driver-assistance systems (ADAS).
Because LidarTLM is less commonly documented and may refer to proprietary or research-centric toolkits, this section will focus on generalized design principles, use cases, and what distinguishes LiDAR-focused runtimes from general-purpose ML engines.
Overview
Developer Target
: Robotics, ADAS, and autonomous system engineers
Use Cases
: Real-time 3D object detection, SLAM (Simultaneous Localization and Mapping), point cloud segmentation, obstacle avoidance
Model Format
: Often custom or adapted from PyTorch/ONNX; serialized as tensors or voxel grids
Conversion Tools
: Typically includes preprocessing pipelines from ROS, Open3D, or custom CUDA kernels
Architecture
LidarTLM-style systems typically deviate from conventional 2D image-based ML runtimes. They require efficient spatial processing, optimized memory layouts, and hardware support for sparse data structures.
Execution Flow
:
Sensor Input
: Raw LiDAR packets or fused multi-sensor data (e.g., IMU + LiDAR) ingested
Preprocessing
: Point clouds downsampled, voxelized, or transformed to Bird’s-Eye View (BEV)
Inference
: Tensorized data passed through neural layers (e.g., 3D convolutions, attention modules)
Postprocessing
: Bounding boxes or semantic maps generated
Fusion (Optional)
: Sensor fusion with radar, camera, or odometry
Key Components
:
Spatial Encoder
: Transforms sparse point clouds into dense tensor formats (e.g., voxel grids, range images)
Sparse CNNs or VoxelNet Layers
: Specialized convolution ops for irregular input data
Temporal Modules
: Optional RNN, attention, or transformer blocks for sequential scans
Hardware Abstraction
: Targets CUDA-enabled GPUs or embedded AI processors (e.g., NVIDIA Xavier, TI Jacinto)
Implementation Details
Tensor Representation
:
Often uses sparse tensors or hybrid dense-sparse structures
Libraries like MinkowskiEngine, SpConv, or custom CUDA kernels for voxel ops
Quantization may be used to reduce memory footprint in embedded settings
Optimization Techniques
:
Efficient neighbor search (KD-trees, octrees) for local feature aggregation
Temporal caching of features from prior scans
Batch fusion for multi-sensor inputs
Deployment
:
Embedded platforms like NVIDIA Jetson, TI DSPs, and ADAS-grade microcontrollers
Often integrated with ROS (Robot Operating System) for I/O and control flow
May use C++, CUDA, or even custom ASIC/NPU firmware for deterministic performance
Pros and Cons
Pros:
Designed for spatial and temporal data, not just 2D tensors
Optimized for sparse inputs and low-latency inference
Supports sensor fusion pipelines, enabling richer context
Can run on edge-grade GPUs or embedded NPUs
Cons:
Fragmented tooling, often bespoke or tightly coupled to hardware
Lack of standardized runtime interface (unlike ONNX or TFLite)
Difficult to deploy across platforms without custom engineering
Sparse community and documentation; often buried in academic or industrial codebases
Example Pseudocode Flow
# Step 1: Load point cloud
point_cloud
=
load_lidar_scan
(
"/scans/frame_001.bin"
)
# Step 2: Convert to voxel grid
voxel_grid
=
voxelize
(
point_cloud
,
grid_size
=
(
0.1
,
0.1
,
0.1
))
# Step 3: Pass through 3D CNN
features
=
sparse_conv_net
(
voxel_grid
)
# Step 4: Predict bounding boxes or labels
detections
=
decode_bounding_boxes
(
features
)
# Step 5: Fuse with other sensors (optional)
fused_output
=
fuse_with_camera
(
detections
,
rgb_frame
)
Suitable Applications
Autonomous vehicles (3D perception stacks)
Warehouse robots and drones
Industrial inspection systems
Advanced driver-assistance systems (ADAS)
SLAM systems for robotics
LidarTLM-like runtimes are not meant for general ML workloads but are highly optimized for 3D spatiotemporal inference, where conventional 2D model runtimes fall short. They tend to be integrated deep into hardware-specific SDKs or research frameworks.
llama.cpp
Deep Dive
llama.cpp
is an open-source, C++-based implementation of inference for large language models (LLMs), originally inspired by Meta’s LLaMA family. It focuses on efficient CPU (and optionally GPU) inference for quantized transformer models. Unlike full ML runtimes,
llama.cpp
is specialized, minimalist, and optimized for running LLMs—particularly on devices with constrained memory and compute budgets such as laptops, desktops, and even smartphones.
Overview
Developer Target
: LLM researchers, app developers, hobbyists
Use Cases
: Local chatbots, privacy-preserving LLM apps, embedded NLP on edge devices
Model Format
: Quantized GGUF (GPT-generated GGML Unified Format)
Conversion Tools
: Python conversion scripts from PyTorch checkpoints to GGUF
Architecture
llama.cpp
does not use a traditional ML runtime stack. It is built from the ground up with custom tensor operations and a static execution loop tailored to transformer inference.
Execution Flow
:
Model Load
: Quantized GGUF file loaded into memory
KV Cache Allocation
: Allocates buffers for key/value attention caching
Token Embedding & Input Prep
: Maps token IDs to embeddings
Layer Execution Loop
: Runs transformer blocks sequentially
Logits Output
: Computes next-token logits, passed to sampler
Sampling & Token Generation
: Greedy, top-k, nucleus, or temperature sampling
Key Components
:
GGML Backend
: Custom tensor library with support for CPU SIMD ops (AVX, FMA, NEON)
Quantization Layers
: 4-bit, 5-bit, and 8-bit quantized matmuls
Inference Loop
: Manually unrolled transformer stack—one layer at a time
KV Cache Management
: Token sequence history for autoregressive decoding
Optional GPU Support
:
Metal (macOS), OpenCL, CUDA support via modular backends
Offloading options: attention only, matmuls only, or full GPU
Implementation Details
Model Quantization
:
Tools like
quantize.py
convert PyTorch models to GGUF format
Supports several quantization strategies (Q4_0, Q5_K, Q8_0, etc.)
Tradeoff between model size and accuracy
Tensor Engine
:
No external libraries like BLAS, cuDNN, or MKL used by default
Uses hand-optimized C++ with platform-specific intrinsics
Cross-platform: macOS, Linux, Windows, WebAssembly (via WASM)
Memory Optimization
:
Memory mapped file support (
mmap
)
Low memory mode: restricts KV cache or context length
Paging and streaming support for large contexts (e.g.,
llama.cpp + vLLM
)
Integration
:
C API and Python bindings (
llama-cpp-python
)
Works with tools like LangChain, OpenRouter, and Ollama
Compatible with most LLaMA-family models: LLaMA, Alpaca, Vicuna, Mistral, etc.
Pros and Cons
Pros:
Extremely fast CPU inference (real-time on MacBook M1/M2, even some Raspberry Pi 4)
Portable and minimal dependencies
Quantization enables running models with <4 GB RAM
Easily embedded into apps, games, and command-line tools
Active community and ecosystem (used in projects like Ollama and LM Studio)
Cons:
Transformer-only; not a general ML runtime
No training support—strictly for inference
Manual conversion and tuning process required
Limited ops support; cannot easily add new ML layers
Example CLI Inference
./main
-m
models/llama-7B.Q4_0.gguf
-p
"What is the capital of France?"
-n
64
Python Inference (via
llama-cpp-python
)
:
from
llama_cpp
import
Llama
llm
=
Llama
(
model_path
=
"llama-7B.Q4_0.gguf"
)
output
=
llm
(
"Q: What is the capital of France?
\n
A:"
,
max_tokens
=
32
)
print
(
output
[
"choices"
][
0
][
"text"
])
WebAssembly Example (Browser):
Precompiled WASM version can run LLMs client-side using WebGPU
Useful for private, offline AI assistants directly in browser
Suitable Applications
Private, offline chatbots
Voice assistants embedded in hardware
Context-aware agents in games or productivity apps
Developer tools with local NLP capabilities
llama.cpp
showcases what is possible with small, optimized transformer runtimes and CPU-centric design. It’s not a general-purpose ML runtime but a powerful engine for language inference where privacy, portability, or internet-free operation is desired.
TensorFlow Lite / TensorFlow Serving Deep Dive
TensorFlow Lite (TFLite) and TensorFlow Serving are two distinct components from the TensorFlow ecosystem optimized for inference, but they serve different purposes and deployment environments.
TensorFlow Lite
is designed for
on-device inference
, particularly for mobile, embedded, and IoT platforms.
TensorFlow Serving
is designed for
cloud and server-side model deployment
, providing high-throughput, low-latency model serving over gRPC or HTTP.
This section focuses primarily on
TensorFlow Lite
due to its relevance to on-device ML runtimes, with a comparative note on Serving at the end.
Overview
Developer Target
: Mobile developers, embedded engineers, production ML ops
Use Cases
: Real-time image classification, object detection, audio processing, NLP, edge analytics
Model Format
:
.tflite
(FlatBuffer format)
Conversion Tools
: TensorFlow \(\rightarrow\) TFLite via
TFLiteConverter
TensorFlow Lite Architecture
TFLite’s design emphasizes performance, size efficiency, and hardware acceleration. It is structured around a model interpreter, a delegate mechanism for hardware acceleration, and a set of optimized operator kernels.
Execution Flow
:
Model Conversion
:
Uses
TFLiteConverter
to convert SavedModel or Keras models into a FlatBuffer-encoded
.tflite
model.
Model Load
:
The model is loaded by the
Interpreter
class on the target device.
Tensor Allocation
:
Memory buffers for input/output tensors are allocated.
Inference Execution
:
The interpreter evaluates the computation graph, optionally using delegates.
Postprocessing
:
Output tensors are read and interpreted by the application.
Key Components
:
FlatBuffer Model
: Compact, zero-copy, serializable model format
Interpreter
: Core engine that evaluates the model graph
Delegate Interface
: Offloads subgraphs to specialized hardware (GPU, DSP, NPU)
Kernel Registry
: Maps ops to optimized C++ implementations (or delegates)
Implementation Details
Model Conversion
:
Converts SavedModels, Keras
.h5
, or concrete functions to
.tflite
Supports post-training quantization (dynamic, full integer, float16)
Model optimizations include constant folding, op fusion, and pruning
Delegates
:
Optional hardware acceleration backends:
NNAPI
(Android)
GPU Delegate
(OpenCL, Metal)
Hexagon Delegate
(Qualcomm DSP)
Core ML Delegate
(iOS/macOS)
EdgeTPU Delegate
(Coral devices)
Delegates work by “claiming” supported subgraphs during interpreter initialization
Threading and Performance
:
Supports multi-threaded inference
Interpreter can be run in C++, Java, Kotlin, Python, Swift
TensorFlow Serving (Short Overview)
Designed for scalable deployment of TensorFlow models on servers
Models are exposed as REST/gRPC endpoints
Automatically loads, unloads, and versions models
Uses
SavedModel
format, not
.tflite
Not suitable for offline or embedded deployment
Use Case Comparison
:
Here is your formatted table following the provided style:
Feature
TensorFlow Lite
TensorFlow Serving
Target Device
Mobile/Edge
Cloud/Server
Model Format
.tflite
SavedModel
Communication
In-process / Local
gRPC / REST
Latency
Milliseconds
Sub-second to seconds
Training Support
No
No (inference only)
Deployment Size
Small (~100s of KB)
Large, server framework
Pros and Cons
Pros (TensorFlow Lite):
Compact and efficient format (FlatBuffer)
Broad hardware delegate support
Quantization-aware and post-training optimizations
Cross-platform support (iOS, Android, Linux, microcontrollers)
Strong ecosystem and pre-trained model zoo (
tflite-model-maker
)
Cons (TensorFlow Lite):
Not a full subset of TensorFlow ops (requires op whitelisting or custom ops)
Delegate behavior can be opaque and platform-dependent
Conversion can fail silently if unsupported ops are encountered
Debugging delegate fallbacks can be non-trivial
Example Inference (Python - TFLite)
import
tensorflow
as
tf
import
numpy
as
np
# Load model
interpreter
=
tf
.
lite
.
Interpreter
(
model_path
=
"mobilenet_v2.tflite"
)
interpreter
.
allocate_tensors
()
# Prepare input
input_details
=
interpreter
.
get_input_details
()
output_details
=
interpreter
.
get_output_details
()
input_data
=
np
.
random
.
rand
(
1
,
224
,
224
,
3
).
astype
(
np
.
float32
)
# Run inference
interpreter
.
set_tensor
(
input_details
[
0
][
'index'
],
input_data
)
interpreter
.
invoke
()
output_data
=
interpreter
.
get_tensor
(
output_details
[
0
][
'index'
])
print
(
"Prediction:"
,
output_data
)
Delegate usage (Android NNAPI, example via Java/Kotlin):
Interpreter
.
Options
options
=
new
Interpreter
.
Options
();
options
.
addDelegate
(
new
NnApiDelegate
());
Interpreter
interpreter
=
new
Interpreter
(
tfliteModel
,
options
);
Suitable Applications
On-device health and fitness apps
Real-time object detection in AR
Offline voice recognition
Edge anomaly detection
TinyML deployments with
TensorFlow Lite for Microcontrollers
TensorFlow Lite remains one of the most production-hardened and flexible runtimes for on-device ML, particularly in mobile and embedded contexts. Its support for multiple delegates and optimizations makes it a go-to choice for developers deploying models outside the cloud.
Comparative Analysis
Here are detailed tabular comparisons that encapsulates all key aspects across the different on-device ML runtimes discussed in the primer.
General Characteristics
Attribute
TensorRT
Core ML
MLX
ONNX Runtime
ExecuTorch
LidarTLM
llama.cpp
TensorFlow Lite
TensorFlow Serving
Target Platform(s)
NVIDIA Jetson, Desktop, Server
Apple devices (iOS/macOS)
Apple Silicon (macOS only)
Cross-platform
Embedded, mobile, MCU
Robotics, automotive, ADAS
Desktop, mobile, browser
Cross-platform (mobile/edge)
Cloud / server environments
ML Task Focus
Optimized inference
General ML (vision, NLP)
Research, transformer/NLP
General ML
Ultra-light inference
3D spatial perception
Large language model inference
General ML
Scalable inference serving
Inference Only?
Yes
Yes
No (supports training)
Yes
Yes
Yes
Yes
Yes
Yes
Open Source?
Partially (binaries open, tools closed)
Partially (via tools)
Yes
Yes
Yes
Partially / variable
Yes
Yes
Yes
Model Formats and Conversion
Attribute
TensorRT
Core ML
MLX
ONNX Runtime
ExecuTorch
LidarTLM
llama.cpp
TensorFlow Lite
TensorFlow Serving
Primary Format
.plan (TensorRT engine file)
.mlmodelc
Python-defined layers
.onnx
.ptc (compiled TorchScript)
Custom / converted .onnx / raw tensors
.gguf (quantized LLMs)
.tflite (FlatBuffer)
SavedModel (.pb, .pbtxt)
Supported Frameworks
PyTorch, ONNX
PyTorch, TF (via converters)
Native Python API
PyTorch, TensorFlow, others
PyTorch (TorchScript subset)
PyTorch, TensorFlow (via export)
LLaMA-family only
TensorFlow, Keras
TensorFlow only
Conversion Required?
Yes (from ONNX or PyTorch export)
Yes (via coremltools)
No
Yes (usually from PyTorch)
Yes (via AOT compiler)
Yes, often includes preprocessing
Yes (convert + quantize)
Yes (TFLiteConverter)
No (already in target format)
Execution Model and Hardware Support
Attribute
TensorRT
Core ML
MLX
ONNX Runtime
ExecuTorch
LidarTLM
llama.cpp
TensorFlow Lite
TensorFlow Serving
Execution Type
AOT compiled CUDA graph
Eager, dynamic hardware assignment
Eager + compiled graph
Static graph with runtime optimizations
Bytecode VM interpreter
Sparse 3D graph + temporal flow
Manual loop over transformer layers
Static interpreter + delegates
REST/gRPC inference pipeline
CPU Support
No (GPU only)
Yes (fallback)
Yes (M1/M2 optimized)
Yes (default EP)
Yes
Yes
Yes (highly optimized)
Yes
Yes
GPU Support
Yes (CUDA, Tensor Cores)
Yes (Metal)
Yes (via MPS)
Yes (CUDA, DirectML, etc.)
Limited
Yes (CUDA, embedded GPUs)
Optional (Metal, CUDA, OpenCL)
Yes (OpenCL, Metal)
No
NPU / DSP Support
No
Yes (Apple ANE)
Emerging ANE support
Yes (via NNAPI, OpenVINO, etc.)
Potential via backend interface
Yes (TI, Nvidia, ADAS accelerators)
No (LLM-focused, CPU-oriented)
Yes (NNAPI, EdgeTPU, Hexagon)
No
Hardware Abstraction
Low-level plugin engine, manual tuning
Automatic
Manual tuning via MLX
Modular Execution Providers (EPs)
Compiled dispatcher with targets
Device-specific optimization required
Low-level SIMD/CUDA offload
Delegate-based (pluggable)
N/A
Optimization, Size, and Constraints
Attribute
TensorRT
Core ML
MLX
ONNX Runtime
ExecuTorch
LidarTLM
llama.cpp
TensorFlow Lite
TensorFlow Serving
Model Optimization Support
Yes (kernel tuning, quantization, `float16`/`int8`)
Yes (ANE targeting, quantization)
No built-in, manual scripting
Yes (quantization, pruning, graph fusion)
Yes (operator pruning, bytecode fusion)
Yes (3D-aware compression and fusions)
Yes (quantized GGUF)
Yes (quantization, fusion)
Yes (batching, threading)
Runtime Size
Medium (~5–15 MB)
Medium (~5–10 MB)
Medium
Large (5–30 MB)
Very small (<1 MB)
Medium–Large
Small–Medium
Small (~0.5–5 MB)
Very large (>100 MB)
Memory Footprint (Inference)
Low to moderate (GPU memory bound)
Low to moderate
Moderate (GPU-heavy)
Variable (depends on EPs)
Ultra-low (sub-MB possible)
High (large point cloud buffers)
Low (~3–6 GB RAM for 7B models)
Low
High
Latency
Very low (sub-ms possible)
Low (with ANE/GPU)
Medium (eager mode)
Variable (highly EP dependent)
Very low
Moderate to high (depends on density)
Low (for small LLMs)
Low (under 10ms typical)
Moderate to high
Flexibility, Debugging, and Ecosystem
Attribute
TensorRT
Core ML
MLX
ONNX Runtime
ExecuTorch
LidarTLM
llama.cpp
TensorFlow Lite
TensorFlow Serving
Custom Ops Support
Yes (via plugin library API)
Limited (via
MLCustomModel
)
Full (via Python subclassing)
Yes (custom EPs and ops)
Yes (C++ op authoring)
Yes (often required)
No (fixed transformer kernel set)
Yes (C++/C custom kernels)
Yes
Community & Documentation
Strong NVIDIA developer support, active forums
Strong, Apple developer-centric
Niche, growing
Very strong
Growing (Meta-sponsored)
Limited / hardware-vendor specific
Active open-source base
Mature, large community
Very mature in production
Debugger Support
Nsight Systems, profiling tools, verbose logging
Xcode tools
Python debug console
Moderate (model inspection tools)
Minimal (CLI, log-based)
Custom tooling per device
Log-level output only
TensorBoard-lite, CLI tools
Monitoring via Prometheus, etc.
Ease of Use
Medium (manual optimization, engine building)
High for Apple developers
Medium (researchers, tinkerers)
Moderate to high (depends on EP)
Medium (steep setup curve)
Low (requires system integration)
High (once model is quantized)
High (especially with
model maker
)
Medium to high (requires infra)
Comparative Summary and Guidance
Feature Comparison Table
This section provides a side-by-side comparison of the on-device ML runtimes discussed, highlighting their architectural differences, platform support, performance characteristics, and ideal use cases. This helps clarify which runtime best fits various project needs, from embedded development to mobile apps and language model inference.
Runtime
Platform Support
Model Format
Hardware Acceleration
Optimized For
Custom Ops
Size Footprint
TensorRT
NVIDIA GPUs (desktop, Jetson, server)
ONNX, `.plan` (engine file)
CUDA, Tensor Cores
Low-latency GPU inference
Yes (via plugin system)
Medium (~5–15 MB)
Core ML
Apple only (iOS/macOS)
`.mlmodelc`
CPU, GPU, ANE
App integration on Apple devices
Limited
Medium (~2–10 MB)
MLX
Apple Silicon (macOS)
Python code
MPS, ANE (partial)
Research & experimentation
Yes
Medium (~2–5 MB)
ONNX Runtime
Cross-platform (Mobile & Desktop)
`.onnx`
CUDA, NNAPI, DirectML, etc.
Cross-framework interoperability
Yes
Large (~5–30 MB)
ExecuTorch
Embedded, MCUs, Android
Compiled TorchScript (`.ptc`)
CPU, MCU, DSP
Ultra-light edge inference
Yes
Very small (<1 MB)
LidarTLM
Embedded/Robotics
Custom/ONNX
CUDA, DSP, NPU
Sparse point cloud inference
Yes
Medium–Large
llama.cpp
Desktop, Mobile, WASM
Quantized GGUF
CPU, Optional GPU
Efficient LLM inference
Limited
Small–Medium (CPU)
TFLite
Cross-platform (MCU to mobile)
`.tflite`
NNAPI, GPU, DSP, EdgeTPU
Mobile and embedded AI
Yes
Small (~500 KB–5 MB)
TF Serving
Cloud/Server
SavedModel
N/A
Scalable online inference
Yes
Very large (>100 MB)
Strengths by Runtime
Core ML
: Best for iOS/macOS developers needing deep system integration with the Apple ecosystem. Ideal for apps that use Vision, SiriKit, or ARKit.
MLX
: Best for Mac-based researchers and developers who want PyTorch-like flexibility and native hardware performance without deploying to iOS.
ONNX Runtime
: Best for cross-platform deployments and teams needing a unified inference backend across mobile, desktop, and cloud. Excellent hardware flexibility.
ExecuTorch
: Best for extremely constrained devices like MCUs, or custom silicon. Perfect for edge intelligence with hard memory and latency budgets.
LidarTLM
: Best for autonomous systems, robotics, and 3D SLAM applications that involve high-bandwidth spatial data like LiDAR or radar.
llama.cpp
: Best for private, local LLM inference on personal devices or embedding transformer models into apps without requiring cloud or heavy runtimes.
TFLite
: Best all-around runtime for mobile and embedded ML. Huge ecosystem, widespread delegate support, and tooling maturity.
TF Serving
: Best for cloud applications needing high-volume model serving (e.g., for APIs). Not designed for local or offline inference.
Runtime Selection Guidance
If you’re deploying to iOS or macOS
:
Use
Core ML
for production apps.
Use
MLX
for research, local experimentation, or custom modeling.
If you’re deploying to embedded edge devices
:
Use
ExecuTorch
for PyTorch-based workflows.
Use
TensorFlow Lite for Microcontrollers
for tight memory constraints.
Consider
LidarTLM
-style tools if dealing with 3D spatial data.
If you’re targeting Android or need portability
:
Use
TensorFlow Lite
or
ONNX Runtime
with delegates like NNAPI or GPU.
If you’re working with LLMs locally
:
Use
llama.cpp
for best CPU-based inference and minimal setup.
If you want cross-framework model portability
:
Use
ONNX Runtime
with models exported from PyTorch, TensorFlow, or others.
If you require real-time, high-volume cloud inference
:
Use
TensorFlow Serving
or ONNX Runtime Server.
Final Thoughts
Choosing the right on-device ML runtime depends heavily on the following factors:
Deployment environment (mobile, embedded, desktop, web, cloud)
Model architecture (CNN, RNN, transformer, etc.)
Performance requirements (latency, FPS, memory usage)
Development preferences (PyTorch, TensorFlow, raw C++, etc.)
Hardware capabilities (CPU, GPU, NPU, DSP, etc.)
Each runtime discussed in this primer is best-in-class for a certain domain or design constraint. Rather than a “one-size-fits-all” solution, success in on-device ML depends on thoughtful matching between the model, target platform, and available tools. Here’s a summary of which is the best runtime across a range of scenarios:
Best for Apple-native app development
:
Core ML
Best for Apple-based model experimentation
:
MLX
Best for cross-platform portability and hardware access
:
ONNX Runtime
Best for minimal embedded inference
:
ExecuTorch
Best for 3D LiDAR/robotics
:
LidarTLM-like stacks
Best for on-device LLM inference
:
llama.cpp
Best for mobile/embedded general ML
:
TensorFlow Lite
Best for scalable cloud inference
:
TensorFlow Serving
Related: Serialization Formats Across Runtimes
In machine learning runtimes, how a model is
serialized
—i.e., stored and structured on disk—is critical for performance, compatibility, and portability. Serialization formats determine how the computation graph, parameters, metadata, and sometimes even execution plans are encoded and interpreted by the runtime. Each runtime typically adopts a format aligned with its optimization goals: whether that’s minimal size, fast loading, platform neutrality, or human-readability for debugging.
Here we briefly compare four major serialization formats used across popular on-device ML runtimes:
Protocol Buffers (Protobuf)
,
FlatBuffer
,
GGUF
, and
Bytecode formats
, reinforcing how data structures are stored, loaded, and interpreted at runtime.
Protocol Buffers (Protobuf)
Used by
: TensorFlow (SavedModel,
.pb
), ONNX (
.onnx
)
Developed by
: Google
Type
: Binary serialization framework
Key Characteristics
:
Encodes structured data using
.proto
schemas
Supports code generation in multiple languages (Python, C++, Java, etc.)
Strict type definitions with schema versioning
Produces portable, efficient, extensible binary files
Advantages
:
Highly compact, faster than JSON/XML
Strong backward and forward compatibility through schema evolution
Ideal for representing complex hierarchical graphs (e.g., model computation trees)
In ML context
:
TensorFlow
: Stores entire computation graph, tensor shapes, and metadata in
.pb
(protobuf binary)
ONNX
: Defines all model ops, weights, and IR-level metadata via Protobuf-defined schema
Limitations
:
Parsing requires full message decoding into memory
Less suited for minimal-footprint scenarios (e.g., MCUs)
Example
:
Used in: TensorFlow (
.pb
, SavedModel), ONNX (
.onnx
)
Protobuf defines a schema in
.proto
files and serializes structured binary data. Here’s a simplified view:
Schema Definition (
graph.proto
):
message
TensorShape
{
repeated
int64
dim
=
1
;
}
message
Node
{
string
op_type
=
1
;
string
name
=
2
;
repeated
string
input
=
3
;
repeated
string
output
=
4
;
}
message
Graph
{
repeated
Node
node
=
1
;
repeated
TensorShape
input_shape
=
2
;
repeated
TensorShape
output_shape
=
3
;
}
Example Python Usage (ONNX-style):
import
onnx
model
=
onnx
.
load
(
"resnet50.onnx"
)
print
(
model
.
graph
.
node
[
0
])
# Shows first operation (e.g., Conv)
Serialized File:
A binary
.onnx
or
.pb
file that’s unreadable in plain text but represents a complete computation graph, including ops, shapes, attributes, and weights.
FlatBuffer
Used by
: TensorFlow Lite (
.tflite
)
Developed by
: Google
Type
: Binary serialization library with zero-copy design
Key Characteristics
:
Allows direct access to data without unpacking (zero-copy reads)
Compact binary representation optimized for low-latency parsing
Built-in schema evolution support
Advantages
:
Near-instantaneous loading—no deserialization overhead
Perfect for mobile/embedded devices with tight latency or startup constraints
Schema-aware tooling for validation
In ML context
:
.tflite
files store computation graphs, tensors, and metadata using FlatBuffer encoding
Facilitates runtime interpretation without converting the graph into a different memory format
Limitations
:
Harder to inspect/debug than JSON or Protobuf
Limited dynamic structure capabilities compared to Protobuf
Example
:
Used in: TensorFlow Lite (
.tflite
)
FlatBuffer does not require unpacking into memory. Instead, the graph is directly accessed as a binary blob using precompiled accessors.
FlatBuffer Schema (simplified):
table Tensor {
    shape: [int];
    type: int;
    buffer: int;
  }

  table Operator {
    opcode_index: int;
    inputs: [int];
    outputs: [int];
  }

  table Model {
    tensors: [Tensor];
    operators: [Operator];
  }
Example Python Usage:
import
tensorflow
as
tf
interpreter
=
tf
.
lite
.
Interpreter
(
model_path
=
"mobilenet_v2.tflite"
)
interpreter
.
allocate_tensors
()
print
(
interpreter
.
get_input_details
())
Serialized File:
A
.tflite
file with FlatBuffer encoding, which includes all tensors, ops, and buffers in an efficient, zero-copy layout.
GGUF (GPT-generated GGML Unified Format)
Used by
:
llama.cpp
and its LLM-compatible ecosystem
Developed by
: Community (successor to GGML model format)
Type
: Lightweight binary tensor format for large language models
Key Characteristics
:
Encodes quantized transformer weights and architecture metadata
Designed for efficient memory mapping and low-RAM usage
Built for CPU-first inference (with optional GPU support)
Advantages
:
Extremely compact, especially with quantization (4–8 bit)
Simple, fast memory-mapped loading (
mmap
)
Compatible with CPU-based inference engines (no dependencies)
In ML context
:
Stores models like LLaMA, Mistral, Alpaca after quantization
Used by
llama.cpp
,
llm.cpp
,
text-generation-webui
, and other local LLM tools
Limitations
:
Not general-purpose—only suitable for transformer LLMs
Lacks complex graph control (branching, dynamic ops)
Example
:
Used in:
llama.cpp
, quantized LLMs*
GGUF (GGML Unified Format) is a binary container for transformer weights and metadata.
Header Block (example layout in binary format):
GGUF
  version: 3
  tensor_count: 397
  metadata:
    model_type: llama
    vocab_size: 32000
    quantization: Q4_0
Python conversion (from PyTorch):
python convert.py
--input
model.bin
--output
model.gguf
--format
Q4_0
Reading from
llama.cpp
:
gguf_context
*
ctx
=
gguf_init_from_file
(
"llama-7B.Q4_0.gguf"
);
ggml_tensor
*
wq
=
gguf_get_tensor_by_name
(
ctx
,
"layers.0.attn.wq"
);
Serialized File:
A
.gguf
file storing quantized tensors, model metadata, and attention layer structure—compact and mmap-compatible.
Bytecode Format (ExecuTorch)
Used by
: ExecuTorch
Developed by
: Meta
Type
: Custom AOT-compiled bytecode
Key Characteristics
:
Outputs compact bytecode (
.ptc
) from PyTorch models via TorchScript tracing
Prunes unused operators to reduce binary size
Embeds minimal op metadata needed for runtime VM
Advantages
:
Highly portable and minimal—can run on MCUs and RTOS platforms
Deterministic memory usage and low overhead
Enables static linking of models and kernels for bare-metal systems
In ML context
:
Targets constrained devices (sub-MB RAM)
Supports fixed operator sets with predictable memory and runtime behavior
Limitations
:
Rigid format—not well suited for dynamic models or rich graph structures
Tied closely to PyTorch tracing and compilation pipeline.
Example
:
Used in: ExecuTorch (
.ptc
format)
ExecuTorch compiles PyTorch models into bytecode similar to a virtual machine instruction set.
Model Compilation:
import
torch
class
Net
(
torch
.
nn
.
Module
):
def
forward
(
self
,
x
):
return
torch
.
relu
(
x
)
scripted
=
torch
.
jit
.
script
(
Net
())
scripted
.
save
(
"net.pt"
)
# TorchScript
# Compile to ExecuTorch format
!
executorchc
compile
--
model
net
.
pt
--
output
net
.
ptc
Runtime Use in C++:
executorch
::
Runtime
runtime
;
runtime
.
load_model
(
"net.ptc"
);
runtime
.
invoke
(
input_tensor
,
output_tensor
);
Serialized File:
A
.ptc
file containing static bytecode for model logic, stripped of unused ops, ready for microcontroller inference.
Comparative Analysis
Understanding the serialization format is crucial when choosing a runtime—especially for performance, portability, and debugging. Developers targeting mobile and embedded environments often prefer FlatBuffer or bytecode for efficiency, while cloud/server or cross-platform projects benefit from Protobuf’s rich graph encoding.
Format
Used By
Format Type
Example File
Viewability
Tool to Inspect
Strengths
Limitations
Protobuf
TensorFlow, ONNX
Binary (schema-driven)
model.onnx
,
model.pb
Binary
onnx
,
tf.saved_model_cli
Cross-platform, schema evolution, rich structure
Larger footprint, full deserialization
FlatBuffer
TensorFlow Lite
Zero-copy binary
model.tflite
Binary
flatc
,
tflite
API
Instant loading, ideal for embedded use
Harder to inspect/debug
GGUF
llama.cpp
Binary tensor map
llama-7B.Q4_0.gguf
Binary
llama.cpp
,
gguf_dump.py
Ultra-compact, mmap-friendly, quantized
LLM-specific only
Bytecode
ExecuTorch
Compiled AOT VM
model.ptc
Binary
executorchc
, ExecuTorch API
Tiny runtime, embedded-friendly
Limited flexibility, PyTorch-only
TensorRT Engine
TensorRT
Binary CUDA engine
model.plan
Binary
TensorRT API (
trtexec
)
Hardware-optimized, precompiled inference
NVIDIA-only, not portable
Model Execution Lifecycle Across ML Runtimes
On-device and edge-focused ML runtimes vary widely in design, hardware support, and internal implementation. However, the overall lifecycle of executing a machine learning model—across any runtime—can be broken down into a common series of stages.
This section provides a deep technical walkthrough of each stage in the lifecycle and sets the foundation for understanding how the specific runtimes (TensorRT, Core ML, MLX, ONNX Runtime, ExecuTorch, LidarTLM, llama.cpp, and TensorFlow Lite/Serving) customize or optimize these stages.
Across all runtimes, model execution follows a common pipeline: convert the trained model into a runtime-compatible format, load and allocate memory, dispatch operations to hardware accelerators (CPU/GPU/NPU), and return structured outputs. Each runtime adapts this flow to its architecture—ranging from compiled CUDA engines (TensorRT) to VM-interpreted bytecode (ExecuTorch) to quantized transformer loops (llama.cpp)—to meet performance, portability, or resource constraints.
General Workflow: From Model to Inference
Model Training
Although training is typically performed in a full ML framework (e.g., PyTorch, TensorFlow), it is critical to know that the trained model must be
exported or converted
into a format compatible with the intended runtime.
This stage outputs:
A trained model file (e.g.,
.onnx
,
.mlmodel
,
.pt
,
.tflite
, etc.)
Associated metadata (input/output shapes, quantization info, etc.)
Model Conversion
This phase adapts the trained model into a runtime-specific format. Conversion tools may also apply
graph simplification
,
quantization
, or
operator fusion
.
Typical tools used:
torch.onnx.export()
(PyTorch \(\rightarrow\) ONNX)
coremltools.convert()
(\(\rightarrow\) Core ML)
TFLiteConverter
(TensorFlow \(\rightarrow\)
.tflite
)
executorchc
(TorchScript \(\rightarrow\) ExecuTorch bytecode)
quantize.py
(for GGUF /
llama.cpp
)
This phase outputs:
Serialized model file tailored for the target runtime
Optional quantized or optimized variant
Model Loading
At runtime, the model file is loaded and deserialized into memory. Runtimes may parse the file into:
Internal intermediate representation (IR)
Execution graph
Bytecode or linear transformer stack (as in
llama.cpp
)
Some runtimes use
zero-copy formats
(e.g., FlatBuffer in TFLite) to avoid overhead.
Memory Allocation
Before inference can occur, the runtime must allocate:
Input and output tensor buffers
Working memory for intermediate computations
(If applicable) KV cache (for LLMs), kernel workspaces, or delegate buffers
Advanced runtimes may precompute memory plans to avoid dynamic allocations (e.g., ExecuTorch,
llama.cpp
).
Inference Execution
The core execution stage involves:
Running the model graph or stack
Dispatching operations (ops) to the appropriate hardware backend (CPU, GPU, NPU)
Managing control flow, caching, and batching
Different runtimes handle scheduling and dispatch differently:
TensorRT: CUDA engine with explicit graph scheduling
TFLite: Static interpreter with delegate hand-off
ONNX Runtime: Execution Providers (EPs)
llama.cpp
: Single-threaded or parallel transformer loop
Postprocessing & Output
The final outputs are:
Raw logits, class probabilities, bounding boxes, text, etc.
Returned via API calls (C++, Python, Swift, etc.)
This stage may also include:
Dequantization
Formatting into app-native types (e.g., Swift structs in Core ML)
Logging and telemetry
Lifecycle Optimization (Optional but Critical)
For deployment, optimization techniques may be inserted at multiple points:
Quantization (during conversion)
Delegate configuration (runtime initialization)
Memory pruning and op fusion (during compile/AOT phase)
Execution profiling and tuning
Runtime-Specific Execution Lifecycles
While the general lifecycle described earlier applies to all runtimes, each ML runtime adapts or specializes this flow to match its architectural goals and supported hardware.
This section provides an
execution lifecycle breakdown
for each runtime discussed in the original primer, with particular focus on runtime-specific logic during model loading, graph execution, memory management, and hardware dispatch.
TensorRT Execution Lifecycle (NVIDIA GPUs)
TensorRT uses an
Ahead-of-Time (AOT) engine-building
process that transforms a model into a highly optimized CUDA execution plan. Once compiled, the
.plan
file encapsulates a pre-fused, quantized, and hardware-specific execution graph.
Lifecycle Stages:
Model Import & Parsing:
Parses ONNX, TensorFlow, or Caffe model using TensorRT parsers.
Builder Optimization:
Applies kernel selection, op fusion,
int8
/
float16
quantization, and layer scheduling.
Engine Generation:
Outputs a
.plan
file containing the serialized CUDA engine.
Runtime Load:
Loads the plan into memory via
IRuntime
, allocates CUDA buffers.
Execution Context:
Prepares
ExecutionContext
with shape bindings, input/output memory views.
Inference Loop:
Launches CUDA kernels via streams with async execution.
Output Retrieval:
Copies GPU output buffers back to host (if needed).
Unique Characteristics:
Extremely low latency, precompiled execution
Requires regeneration if model shape changes
All ops dispatched on GPU only
Core ML Execution Lifecycle (Apple Platforms)
Core ML performs inference via
runtime graph execution
of a compiled
.mlmodelc
package. It abstracts backend selection and heavily integrates with Apple’s APIs.
Lifecycle Stages:
Model Compilation:
.mlmodel
\(\rightarrow\)
.mlmodelc
via Xcode or
coremltools
App Initialization:
Loads model via
MLModel(configuration:)
Backend Dispatch:
Chooses CPU, GPU, or ANE depending on hardware availability and op support.
Inference Call:
model.prediction(input:)
executes internal graph
Result Handling:
Outputs are returned as native Swift types (e.g., strings, arrays, dicts)
Unique Characteristics:
Dynamic backend selection with op-level granularity
Opaque execution graph, no public access to IR
Secure, sandboxed memory isolation for inference
MLX Execution Lifecycle (Apple Silicon)
MLX uses a
Python-based tensor programming model
and optionally compiles graphs via JIT. It is most similar to PyTorch but tightly integrated with Metal.
Lifecycle Stages:
Model Definition:
Model is defined in Python using
mlx.nn.Module
Eager Execution (default):
Runs ops immediately using Metal Performance Shaders (MPS)
Compiled Graph (optional):
@mlx.compile
transforms a function into a static kernel sequence
Tensor Handling:
All tensors are immutable; memory reuse is managed by the MLX runtime
Execution:
Kernel invocations are dispatched via Metal; ANE support is under development
Output:
Results returned as MLX tensors, convertible to NumPy or PyTorch
Unique Characteristics:
Developer-centric and Pythonic
Targets M1/M2 GPU via Metal
No external model serialization—code
is
the model
ONNX Runtime Execution Lifecycle
ONNX Runtime is built around an
intermediate computation graph
, modular kernel registry, and
Execution Providers (EPs)
that delegate ops to appropriate hardware.
Lifecycle Stages:
Model Load:
Parses
.onnx
file (protobuf format) into IR
Graph Optimization:
Applies passes (e.g., constant folding, op fusion, node elimination)
EP Assignment:
Ops are split across available EPs (CPU, CUDA, NNAPI, etc.)
Session Initialization:
Prepares
InferenceSession
with input/output bindings
Execution:
Each partition of the graph is dispatched to its EP
Result Aggregation:
Output tensors are collected and returned in native types
Unique Characteristics:
Pluggable backend system for flexible hardware support
Static graph, dynamic shape support with constraints
Strong cross-platform model portability
ExecuTorch Execution Lifecycle (MCU/Embedded Focus)
ExecuTorch employs a
bytecode VM
model with AOT compilation for PyTorch models. It is built for microcontrollers and embedded edge devices.
Lifecycle Stages:
TorchScript Compilation:
PyTorch model scripted and converted into
.pt
(TorchScript)
AOT Bytecode Generation:
executorchc
compiles model to
.ptc
(bytecode)
Runtime Embedding:
Bytecode and interpreter embedded into firmware or C++ app
Interpreter Loop:
Model execution performed by a tiny VM that reads bytecode
Op Dispatch:
Ops are routed to statically compiled function pointers
Output Return:
Inference results written to statically allocated output buffer
Unique Characteristics:
Deterministic memory, static allocation only
Supports sub-MB runtime environments
Highly tunable; model format ≠ PyTorch IR
LidarTLM Execution Lifecycle (LiDAR-Focused Embedded Stacks)
LidarTLM-style runtimes are not general-purpose, but highly optimized for 3D spatial inference using sparse tensor pipelines.
Lifecycle Stages:
Sensor Input:
LiDAR frames streamed in real-time
Preprocessing:
Voxelization or range transformation into tensor-friendly formats
Tensor Pipeline:
Sparse CNNs, 3D convolutions, and attention modules process data
Temporal Fusion:
RNN or transformer-based modules optionally applied across frames
Postprocessing:
Generates semantic maps or bounding boxes
Sensor Fusion:
Optionally integrates radar or camera data for final outputs
Unique Characteristics:
Sparse tensors and voxel grids dominate memory model
CUDA, Open3D, or MinkowskiEngine often used
Hard real-time constraints for robotics/ADAS
llama.cpp
Execution Lifecycle (Quantized LLMs)
llama.cpp
is a minimalist CPU-first runtime for LLMs using quantized models in the GGUF format. It has no graph engine—just a static transformer loop.
Lifecycle Stages:
Model Load:
GGUF model memory-mapped into RAM
KV Cache Setup:
Pre-allocates attention buffers
Embedding \(\rightarrow\) Transformer Loop:
Sequentially executes transformer layers
Sampling:
Next token is selected via greedy/top-k/top-p logic
Tokenization:
Output string is constructed from sampled token IDs
Unique Characteristics:
Highly portable, CPU-optimized, extremely low memory usage
No dynamic graph, no scheduler, no intermediate representation
Offload options (e.g., Metal, CUDA) are modu#### TensorFlow Lite Execution Lifecycle
TensorFlow Lite Execution Lifecycle
TFLite uses a
FlatBuffer interpreter
architecture with optional delegates for acceleration.
Lifecycle Stages:
Model Conversion:
TensorFlow \(\rightarrow\)
.tflite
via
TFLiteConverter
FlatBuffer Load:
Model loaded with
Interpreter(model_path=...)
Tensor Allocation:
Input/output buffers allocated via
allocate_tensors()
Delegate Attachment (optional):
NNAPI, GPU, Hexagon delegate claims subgraphs
Inference:
Static interpreter walks the computation graph
Output Access:
Results extracted via
get_tensor()
APIs
Unique Characteristics:
Very compact format with zero-copy access
Delegate design separates concerns for CPU vs. accelerators
Strong ecosystem with tooling (e.g., Model Maker, Visualizer)
Related: CPU Operator Libraries/Backends
FBGEMM
,
QNNPACK
, and
XNNPACK
are high-performance CPU operator libraries used by runtimes (and frameworks embedding a runtime) to execute model operators efficiently.
Note that quantization is performed by the framework either during inference (dynamic), training-time (QAT), or post-training (PTQ), producing per-tensor or per-channel scales/zero-points \((s, z)\) or \((s_c, z_c)\) via the affine map \(x_q=\mathrm{clip}_{[q_{\min},\,q_{\max}]}(\mathrm{round}(x/s)+z)\) with dequantization \(x\approx s\,(x_q-z)\). These backends (
FBGEMM
,
QNNPACK
, and
XNNPACK
) do not perform quantization but rather consume these parameters and the resulting low-precision tensors (or unquantized floating-point tensors for XNNPACK paths) to run inference kernels efficiently.
FBGEMM
targets x86/servers with fast
int8
GEMM/conv;
QNNPACK
targets ARM/mobile CPUs with optimized
int8
conv/GEMM/activations;
XNNPACK
focuses on fast
float32
kernels with some
int8
paths and commonly handles float ops.
Overview
This quick overview shows how
FBGEMM
,
QNNPACK
, and
XNNPACK
fit into the CPU inference stack—what they do, what they don’t, and when runtimes route ops to them:
Primary purpose
: Efficient execution of CPU kernels (conv/GEMM, depthwise conv, elementwise ops) for float and/or 8-bit quantized tensors.
They do not
: Choose quantization parameters, calibrate ranges, or convert weights—framework/tooling does that.
Where they fit
: After model conversion and quantization, the runtime dispatches supported ops to one of these libraries based on platform and data type.
FBGEMM
(by Meta,; Server CPUs)
Target platforms
: x86-64 server/desktop CPUs with SIMD (AVX2, AVX-512; newer stacks can leverage AMX on recent Intel parts via higher-level integrations).
Data types and quant schemes
:
int8
/
uint8
activations (affine per-tensor),
int8
weights (often symmetric per-channel). 32-bit accumulation with requantization to
int8
/
uint8
or dequantization to float. Also provides row-wise and 4-bit embedding quantization utilities for recommendation models.
Operator coverage
: Linear/GEMM and convolution (including groupwise), prepacked weight paths; optimized im2col/IGEMM; embedding bag and sparse length ops for recsys.
Optimizations
: Weight pre-packing into cache-friendly blocked layouts; vectorized micro-kernels; cache- and register-blocking; fused bias+activation+requant paths; threadpool parallelism.
Typical use
: PyTorch quantized ops on server/desktop CPUs (e.g., dynamic quantized Linear/LSTM, static
int8
conv/linear). Best when you need maximum x86 performance for
int8
inference.
; Mobile CPUs)
Target platforms
: ARM/ARM64 mobile CPUs with NEON (Android/iOS); designed for low-power cores.
Data types and quant schemes
:
uint8
/
int8
activations (affine per-tensor),
int8
per-channel weights; 32-bit accumulation with efficient requantization.
Operator coverage
: Quantized convolution/IGEMM, depthwise conv, deconvolution, fully connected (GEMM), pooling, various activation/elementwise ops.
Optimizations
: NHWC-friendly kernels; careful cache use for small batch/small filters; per-thread micro-kernels; fused post-ops to reduce memory traffic.
Typical use
: PyTorch Mobile’s quantized back end on ARM; good default for mobile
int8
CNNs and fully connected layers where you need predictable latency on phones.
XNNPACK
(by Google,; both Server and Mobile CPUs)
Target platforms
: ARM/ARM64, x86-64, and WebAssembly (WASM); broadly portable and actively maintained.
Data types and quant schemes
: Strong
float32
/
float16
/
bfloat16
coverage; mature QS8/QU8 (signed/unsigned 8-bit) inference for conv/GEMM/elementwise with per-channel weight scales. 32-bit accumulation and precise requantization.
Operator coverage
: Convolution (standard and depthwise), fully connected, pooling, deconvolution, elementwise math, softmax, activation functions, resize, etc.
Optimizations
: Handwritten micro-kernels per ISA (NEON/AVX/AVX512), NHWC dataflow, weight prepacking, GEMM/IGEMM families with cache-aware blocking, parallel work-stealing.
Typical use
: TensorFlow Lite’s XNNPACK delegate on CPU (float and
int8
), and increasingly as a CPU backend in other frameworks for both float and quantized inference.
What to Choose When?
PyTorch (desktop/server CPU)
:
FBGEMM
is the usual backend for quantized ops; dynamic quantized Linear/LSTM also route here.
PyTorch Mobile (ARM)
:
QNNPACK
is the historical default for quantized ops; some float operators can use XNNPACK.
TensorFlow Lite (CPU)
:
XNNPACK
delegate accelerates many
float32
and
int8
ops; the interpreter falls back to reference kernels when needed.
ONNX Runtime (CPU)
: Uses its own CPU kernels by default, but can be built/integrated with these libraries in certain configurations; on mobile, builds commonly leverage
XNNPACK
.
Design Notes
Quant params are part of tensors
: Kernels need correct scales/zero-points. For per-channel weights, pass channel-wise scales; activations are usually per-tensor.
Accumulation width
: 8-bit multiply-accumulates are summed into 32-bit accumulators to avoid overflow, then requantized. Watch for saturation when chaining ops.
Prepack once
: Pre-pack and reuse weights to avoid paying packing costs per inference. Many APIs expose prepacked weight objects.
Layout matters
: These libraries typically prefer NHWC for conv on mobile; mismatched layouts cause costly transposes.
Dynamic vs static quant
: Dynamic quantizes activations on-the-fly (common for Linear/LSTM), static uses calibration ranges. FBGEMM has strong dynamic Linear/LSTM paths.
Activation ranges
: Calibrate representative inputs to pick good scales and avoid clamp-heavy requantization.
Comparative Analysis
Attribute
FBGEMM
QNNPACK
XNNPACK
Primary target
x86-64 servers/desktops
ARM/ARM64 mobile
ARM/ARM64, x86-64, WASM
Best precision
`int8` quant (server)
`int8` quant (mobile)
`float32`/`float16` plus `int8`
Typical consumers
PyTorch quant (server)
PyTorch Mobile quant
TFLite delegate; some PyTorch CPU paths
Conv layout
NCHW/NHWC with prepack
NHWC
NHWC
Weight handling
Prepacked per-channel `int8`
Prepacked per-channel `int8`
Prepacked per-channel `int8`
Further Reading
Efficient Inference with Transformer Models on CPUs
Speculative Decoding for Accelerated Transformer Inference
Fast Transformers with Memory-Efficient Attention via KV Cache Optimization
SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
Intel Extension for PyTorch: Boosting Transformer Inference on CPUs
FasterTransformer GitHub Repository (NVIDIA)
vLLM: Easy and Fast LLM Serving with State-of-the-Art Throughput
Deploying Transformer Models on Edge Devices with TensorRT
Quantization Aware Training in PyTorch
ONNX Runtime: Accelerating Transformer Inference
Speculative Decoding in vLLM (Medium article)
Running LLMs on Mobile: Lessons from Distilling and Quantizing GPT-2
Optimizing LLM Serving on NVIDIA GPUs with TensorRT-LLM
LLM INT4 Inference with ONNX Runtime
Efficient Transformer Inference on Edge with EdgeTPU
IREE: Intermediate Representation Execution Environment
XLA
StableHLO
Citation
If you found our work useful, please cite it as:
@article{Chadha2020DistilledMLRuntimes,
  title   = {ML Runtimes},
  author  = {Chadha, Aman},
  journal = {Distilled AI},
  year    = {2020},
  note    = {\url{https://aman.ai}}
}