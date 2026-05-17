# GPU Architecture

**Source**: https://aman.ai/primers/ai/gpu-architecture
**Category**: Miscellaneous
**Word Count**: 3770

---

Overview
Fundamental Architectural Components
Streaming Multiprocessors (SMs)
CUDA Cores
Tensor Cores
Ray Tracing (RT) Cores
Memory Hierarchy
Interconnects
Key Architectural Design Goals
Execution Paradigms in NVIDIA GPUs
Thread and Warp Model
Thread Blocks and Grids
SIMT Execution Model
Warp Scheduling
Synchronization and Communication
Workflow: Running AI Models on NVIDIA GPUs
Model Definition and Training (CPU/GPU)
Operator Dispatch and Kernel Mapping
Memory Management and Transfer
Kernel Launch and Execution on SMs
Forward and Backward Passes (Training)
Inference Deployment
Precision Optimization
Runtime Tools and Profiling
Compute Architecture Evolution
Streaming Multiprocessors (SM) Evolution
CUDA Core Advancements
Tensor Core Evolution
Ray Tracing Core Evolution
Floating-Point Precision Performance Evolution
Overview of Precision Types
Per-Generation Precision Support and Performance
Performance Analysis
Memory Architecture Evolution
VRAM Technologies and Bandwidth
Cache Hierarchy Changes
Memory Latency and Efficiency Improvements
Multi-GPU Memory Coherency
Comparative Analysis of NVIDIA Architectures
AMD GPU Architecture Overview and Comparison with NVIDIA
Fundamental Architectural Components (AMD)
Execution Paradigms (AMD)
Compute Architecture Evolution (AMD CDNA & RDNA)
Floating-Point Precision Performance Evolution
Memory Architecture Evolution
Comparative Analysis: AMD vs. NVIDIA Architectures
Key Takeaways
Citation
Overview
Graphics Processing Units (GPUs) are massively parallel processors originally designed for rendering graphics but now widely used for general-purpose high-throughput computing. Their architecture is optimized for workloads that can be parallelized, such as:
Graphics rendering (rasterization, shading, ray tracing)
Scientific computing
Machine learning training and inference
Simulation and data analytics
Unlike CPUs, which prioritize low-latency execution of serial tasks, GPUs focus on high-throughput execution of many threads simultaneously.
Fundamental Architectural Components
Streaming Multiprocessors (SMs)
The SM is the fundamental execution unit in an NVIDIA GPU. Each SM contains:
Multiple CUDA cores (integer and floating-point ALUs)
Special Function Units (SFUs) for transcendental math
Tensor Cores for matrix-heavy operations (deep learning, HPC)
Load/store units for memory access
Warp schedulers for issuing instructions to warps
CUDA Cores
Handle general-purpose floating-point and integer operations.
Organized into warps of 32 threads.
Designed for SIMD-like execution, where all threads in a warp execute the same instruction.
Tensor Cores
Specialized hardware for matrix multiply-and-accumulate (MMA) operations.
Key for AI/ML workloads (e.g., FP16, BF16, INT8 precision).
Introduced in Volta, improved in each subsequent architecture.
Ray Tracing (RT) Cores
Accelerate BVH traversal and ray-triangle intersection tests for real-time ray tracing.
First introduced in Turing architecture.
Memory Hierarchy
NVIDIA GPUs employ a multi-tier memory system:
Registers
(per thread, lowest latency)
Shared Memory / L1 Cache
(per SM, low latency, user-controlled or cache mode)
L2 Cache
(shared across all SMs, higher capacity)
Global Memory (VRAM)
(off-chip, high latency, GDDR6 or HBM)
Texture and constant caches
(specialized caching units)
Interconnects
NVLink
: High-bandwidth, low-latency GPU-to-GPU and GPU-to-CPU interconnect.
PCIe
: Standard system interconnect, slower than NVLink.
Key Architectural Design Goals
High Parallelism
– Scaling SM count and CUDA cores.
Specialization
– Adding RT and Tensor Cores for dedicated workloads.
Memory Bandwidth
– Using faster VRAM and wider buses.
Energy Efficiency
– Improved performance per watt through better fabrication processes and architectural optimizations.
Scalability
– Supporting multi-GPU setups with coherent memory models.
Execution Paradigms in NVIDIA GPUs
Execution in NVIDIA GPUs follows a
hierarchical, SIMT-based
(Single Instruction, Multiple Threads) model. Understanding this hierarchy is crucial before comparing how specific architectures (Ampere \(\rightarrow\) Blackwell) evolved.
Thread and Warp Model
Thread
:
The smallest unit of execution. Each thread has its own registers, program counter, and can execute independently, but in practice, threads are grouped for efficiency.
Warp
:
A warp is a group of
32 threads
that execute the same instruction in lockstep on a single SM.
If threads within a warp diverge (take different branches), execution is serialized for each path until they reconverge (warp divergence penalty).
Warps are the main scheduling unit in NVIDIA GPUs.
Thread Blocks and Grids
Thread Block (also known as Cooperative Thread Array)
:
A thread block, also called a Cooperative Thread Array (CTA), is a group of threads (typically up to 1024) organized in 1D, 2D, or 3D layouts that work together, sharing data via shared memory and synchronizing their execution.
Threads in a block share
SM-local resources
such as shared memory and L1 cache.
A block is always executed on a single SM (cannot span multiple SMs).
Grid
:
A grid is a set of thread blocks launched for a kernel execution.
Can have 1D, 2D, or 3D configuration.
Enables scaling up to millions of threads.
SIMT Execution Model
SIMT is similar to SIMD but with more flexibility—each thread has its own instruction state, yet warps execute in a vector-like fashion.
Hardware warp schedulers select ready warps each cycle to hide memory latency.
The SIMT model allows massive parallelism while tolerating high memory access latencies.
Warp Scheduling
Each SM contains
warp schedulers
(number varies by architecture).
The scheduler chooses one or more ready warps per cycle, issuing instructions to execution units.
Techniques to maximize utilization:
Latency hiding
: Switching to another ready warp when one stalls.
Dual-issue
: Issuing two independent instructions in the same cycle.
Specialized pipelines
: Routing instructions to CUDA cores, Tensor Cores, or SFUs.
Synchronization and Communication
Intra-warp
: Implicit, as all threads execute in lockstep.
Intra-block
: Achieved via
__syncthreads()
barrier and shared memory.
Inter-block
: No built-in sync; requires multiple kernel launches or cooperative groups.
Workflow: Running AI Models on NVIDIA GPUs
Deep learning frameworks like PyTorch and TensorFlow offer high-level APIs to define, train, and deploy models. When these models are executed on NVIDIA GPUs, a well-defined sequence of steps is followed to convert Python-based model definitions into GPU-executable workloads. Below is a detailed breakdown of this workflow, focusing on how model computations map to GPU architecture.
Running AI models on NVIDIA GPUs involves several coordinated layers: the high-level model definition, dispatch of operators to highly optimized CUDA kernels, memory transfers and execution on SMs, and runtime profiling. Modern GPUs—with their hierarchical memory, massive parallelism, and hardware-accelerated units like Tensor Cores—are architected to accelerate this end-to-end workflow. The software stack abstracts much of the complexity, enabling researchers and engineers to deploy models with minimal hardware-level code but with deep performance leverage.
Model Definition and Training (CPU/GPU)
Models are typically defined in Python using high-level APIs in PyTorch (
nn.Module
) or TensorFlow (
tf.keras.Model
). During training, operations like matrix multiplications, convolutions, and non-linearities are recorded dynamically (in PyTorch’s eager mode) or statically (in TensorFlow’s graph mode).
If a GPU is available and selected (
device="cuda"
or
tf.device('/GPU:0')
), the framework schedules these computations for GPU execution.
Operator Dispatch and Kernel Mapping
Each tensor operation (e.g.,
matmul
,
ReLU
,
conv2d
) corresponds to one or more GPU kernels. The deep learning framework uses backend libraries like:
cuDNN
(for deep neural networks: convolutions, activation, normalization)
cuBLAS
(for matrix algebra)
cuFFT
,
cuSPARSE
, etc. (for specialized ops)
These libraries provide pre-optimized CUDA kernels tailored for various tensor shapes and precisions (e.g., FP32, FP16, BF16, INT8). When a model executes, the framework dispatches operator calls to these GPU-optimized kernels via these libraries.
Memory Management and Transfer
Before execution, model parameters and inputs are moved to GPU memory (VRAM). PyTorch and TensorFlow both manage memory allocation, deallocation, and device transfers:
CPU to GPU:
tensor.to("cuda")
(PyTorch) or
.gpu()
context (TensorFlow)
GPU memory is allocated dynamically and reused to reduce fragmentation
Intermediate results during forward/backward passes are cached in GPU RAM to avoid excessive recomputation
Efficient memory use is critical, especially for large models and high-resolution inputs.
Kernel Launch and Execution on SMs
Once data and operations are prepared, each operator triggers a
CUDA kernel launch
. This is where NVIDIA’s execution model comes into play:
Kernels are executed by
Streaming Multiprocessors (SMs)
, leveraging thousands of
CUDA cores
Work is parallelized into
warps
(32 threads), grouped into
thread blocks
Tensor Cores accelerate operations like GEMMs and convolutions in mixed precision (FP16, BF16, TF32)
The GPU’s warp schedulers manage kernel instruction dispatch, hide memory latency, and maximize throughput by overlapping compute and memory-bound operations.
Forward and Backward Passes (Training)
Forward pass
: Model computations are executed layer by layer, and activations are stored.
Backward pass
: Gradients are computed using automatic differentiation, with additional kernels launched for each operation’s derivative.
Gradient updates
: Optimizer steps (e.g., Adam, SGD) are also dispatched to GPU kernels when tensors reside in GPU memory.
All of this happens in the context of a training loop, with heavy reliance on GPU compute resources and memory bandwidth.
Inference Deployment
Once trained, the model is often exported for inference:
TensorFlow:
SavedModel, TF Lite, or ONNX format
PyTorch:
TorchScript, ONNX, or native PyTorch weights
During inference, only the forward pass is executed. Frameworks and inference engines (like TensorRT, TorchServe, or TF Serving) strip unnecessary training components, optimize kernel ordering, and preload weights into GPU memory to reduce latency and maximize throughput.
Precision Optimization
To reduce GPU memory usage and increase performance, models often use:
Mixed precision training
(e.g.,
float16
for activations,
float32
for loss and gradients)
Quantized inference
(e.g.,
int8
for weights and activations)
These techniques rely on the GPU’s Tensor Cores and require calibration or loss-scaling strategies for training stability.
Runtime Tools and Profiling
Execution can be monitored and optimized using:
Nsight Compute/Systems
: Low-level GPU profiling
TensorBoard
: Graph inspection and timing (TensorFlow)
PyTorch Profiler
: Operator-level breakdown and bottleneck tracing
NVIDIA’s CUPTI
: Provides hooks for collecting runtime metrics
Compute Architecture Evolution
Streaming Multiprocessors (SM) Evolution
Ampere (2020)
:
SMs housed 128 CUDA cores (vs. 64 in Turing).
Warp schedulers improved to issue instructions to mixed precision units simultaneously.
Dual datapath per CUDA core allowed FP32 + INT32 execution concurrently.
FP64 throughput doubled in data center models (A100).
Hopper (2022)
:
Redesigned SMs for higher clock speeds and more instruction-level parallelism.
Added
DPX instructions
for dynamic programming acceleration (bioinformatics, optimization).
Warp specializations for matrix workloads to better feed Tensor Cores.
Increased register file size for HPC workloads.
Ada Lovelace (2022)
:
Focused on gaming and creative workloads.
Higher boost frequencies per SM.
Improved power gating for efficiency.
Enhanced scheduling for real-time ray tracing workloads.
Blackwell (2024)
:
Next-generation SMs with
thread block clusters
for improved multi-SM cooperation.
Improved simultaneous multi-kernel execution.
Expanded warp schedulers to reduce instruction stalls in large AI inference.
Further increased FP8, BF16, and mixed-precision throughput in SM datapaths.
CUDA Core Advancements
Ampere:
128 CUDA cores/SM, concurrent FP32 and INT32 execution per clock.
Hopper:
Improved dual-issue scheduling, better cache locality for CUDA workloads.
Ada:
Boosted per-core clock frequency; targeted gaming rasterization and shading throughput.
Blackwell:
Higher per-core IPC, deeper pipelines for AI workloads.
Tensor Core Evolution
Generation
Precision Support
Notable Features
Ampere
float16
,
bfloat16
,
TensorFloat-32
,
int8
,
int4
TF32 introduced for AI training; structured sparsity (2:4 pattern).
Hopper
Adds
float8
, improved
bfloat16
/
float16
Transformer Engine for mixed-precision AI acceleration.
Ada
Similar to Hopper for consumer SKUs
AI super resolution for DLSS 3.x.
Blackwell
float4
, expanded
float8
Enhanced Transformer Engine for LLMs, better sparsity handling.
Ray Tracing Core Evolution
Ampere:
2nd-gen RT Cores, hardware triangle intersection, motion blur support.
Hopper:
Primarily HPC focus; RT improvements not the priority.
Ada:
3rd-gen RT Cores, Opacity Micromaps, Displaced Micro-Meshes.
Blackwell:
AI-assisted ray traversal prediction, further pipeline efficiency gains.
Floating-Point Precision Performance Evolution
This section focuses on the supported precision formats, their throughput improvements, and architectural features that directly impact floating-point performance across NVIDIA’s recent architectures.
Overview of Precision Types
NVIDIA GPUs handle multiple floating-point formats for different workloads:
FP64 (Double Precision):
64-bit IEEE format; essential for scientific and HPC workloads.
FP32 (Single Precision):
32-bit IEEE format; common for graphics, simulations, and ML training.
TF32 (Tensor Float 32):
19-bit precision (8-bit exponent, 10-bit mantissa) introduced in Ampere for AI workloads—maintains FP32 range with reduced precision for faster computation.
FP16 (Half Precision):
16-bit IEEE format; used for mixed-precision deep learning training.
BF16 (Brain Floating Point 16):
16-bit format with FP32’s exponent range but reduced mantissa precision; popular in AI training for stability.
FP8:
8-bit floating point (two formats E4M3 and E5M2); optimized for AI inference.
FP4:
4-bit floating point; introduced in Blackwell for ultra-low precision inference.
Per-Generation Precision Support and Performance
Ampere (2020)
:
FP64: 1/2 rate of FP32 in A100 (HPC models), 1/64 in consumer cards.
FP32: 2× throughput vs Turing; dual datapath allowed FP32 + INT32 execution.
TF32: Enabled on Tensor Cores; up to 20× faster AI training vs pure FP32.
FP16/BF16: Tensor Core acceleration, ~312 TFLOPS on A100.
INT8/INT4: Supported with sparsity acceleration.
Hopper (2022)
:
FP64: Maintained 1/2 rate in HPC variants (H100).
FP32: Slight IPC and clock improvements.
FP16/BF16: Transformer Engine dynamically chose precision (FP16, BF16, FP8) per layer to optimize throughput.
FP8: Introduced with up to 4× throughput improvement over FP16 for inference.
TF32: Same as Ampere but with efficiency gains via scheduling.
Ada Lovelace (2022)
:
FP64: Mostly absent in consumer models (used in some pro variants at reduced rate).
FP32: Highest per-core clock speeds to date for rasterization.
FP16/BF16: Similar to Hopper in supported formats but tuned for DLSS and AI upscaling.
FP8: Present in professional Ada GPUs for AI workloads.
Blackwell (2024)
:
FP64: Maintains HPC capability at 1/2 FP32 rate in B100/B200.
FP32: Further IPC gains; optimized for large-scale AI as well as HPC.
FP16/BF16: Higher throughput than Hopper with enhanced Transformer Engine.
FP8: Doubled throughput over Hopper for inference.
FP4: New ultra-low precision mode; allows extremely high throughput for LLM inference with minimal accuracy loss using quantization-aware training.
Performance Analysis
Generation
FP64 (HPC)
FP32
TF32
FP16/BF16
FP8
FP4
Ampere
1/2 FP32 rate (A100)
Dual datapath, +2× Turing
Yes
Tensor Core, ~312 TFLOPS
No
No
Hopper
1/2 FP32 rate (H100)
IPC + clock boost
Yes
Transformer Engine
Yes
No
Ada
Limited
High clocks
Yes
Consumer AI workloads
Some pro models
No
Blackwell
1/2 FP32 rate (B100/B200)
Higher IPC
Yes
Enhanced Transformer Engine
Yes, 2× Hopper
Yes
Memory Architecture Evolution
This section examines changes in
VRAM type
,
memory bandwidth
,
cache hierarchy
, and
new memory-related technologies
introduced across NVIDIA architectures from Ampere through Blackwell.
VRAM Technologies and Bandwidth
Ampere (2020)
:
Data center (A100):
Used
HBM2e
with up to 1.6 TB/s bandwidth.
Consumer (RTX 30-series):
Used
GDDR6X
(developed with Micron) on higher-end cards for >900 GB/s bandwidth (RTX 3090).
Bus widths:
Up to 384-bit on flagship consumer GPUs.
Hopper (2022)
:
H100:
Used
HBM3
in some SKUs, offering up to 3 TB/s bandwidth.
Support for larger memory capacities per GPU package (up to 80 GB HBM3).
Designed for massive AI model parameter storage in-memory.
Ada Lovelace (2022)
:
Consumer focus:
GDDR6X on high-end cards, GDDR6 on midrange.
Bandwidth efficiency improved with
L2 cache enlargement
, reducing VRAM fetch pressure.
Top models (RTX 4090) reached 1 TB/s effective bandwidth.
Blackwell (2024)
:
Data center (B100/B200):
HBM3e with >4 TB/s bandwidth in top configurations.
Consumer:
Higher-speed GDDR7 for >1.2 TB/s bandwidth on enthusiast cards.
Improved memory controllers for lower latency in AI workloads.
Cache Hierarchy Changes
Ampere
:
L2 cache sizes: up to 40 MB on A100, smaller on consumer (~6 MB).
Configurable L1/shared memory up to 192 KB per SM.
Hopper
:
L2 cache: 50 MB on H100.
L1/shared memory bandwidth doubled compared to Ampere.
Improved cache coherence across NVLink-connected GPUs.
Ada
:
Significantly larger L2 cache for consumer GPUs (72 MB on RTX 4090).
Reduced VRAM dependence for gaming workloads.
Blackwell
:
Unified large L2 (100+ MB) for data center GPUs.
Faster shared memory with AI-aware prefetching.
AI-managed cache policies to keep transformer weights resident.
Memory Latency and Efficiency Improvements
Ampere:
Introduced
structured sparsity
support in Tensor Cores, reducing VRAM traffic.
Hopper:
Added
Transformer Engine
, dynamically choosing precision to reduce memory footprint.
Ada:
Focused on L2 cache expansion to mask VRAM latency.
Blackwell:
Integrated
streaming memory partitioning
for large model inference; data preloaded into SM-local caches before execution.
Multi-GPU Memory Coherency
Ampere:
NVLink 3.0 with 600 GB/s bidirectional bandwidth, partial memory coherency.
Hopper:
NVLink 4.0 with 900 GB/s bandwidth, full memory coherency across up to 256 GPUs in NVSwitch topologies.
Ada:
NVLink limited or absent in consumer cards.
Blackwell:
NVLink 5.0, >1 TB/s, improved for multi-node AI training with direct GPU-to-GPU streaming.
Comparative Analysis of NVIDIA Architectures
Feature
Ampere (2020)
Hopper (2022)
Ada Lovelace (2022)
Blackwell (2024)
Process Node
Samsung 8N (consumer), TSMC 7N (data center)
TSMC 4N
TSMC 4N
TSMC 3N
SM Design
128 CUDA cores/SM, dual datapath (FP32 + INT32)
Enhanced ILP, DPX instructions, warp specialization
Higher clocks, improved RT workload scheduling
Thread block clusters, expanded warp schedulers
CUDA Cores
Up to 10,752 (A100)
Up to 16,896 (H100)
Up to 16,384 (RTX 4090)
>20,000 in top HPC SKUs
Tensor Cores
3rd-gen: FP16, BF16, TF32, INT8, INT4
4th-gen: adds FP8, Transformer Engine
Similar to Hopper for pro models
5th-gen: adds FP4, better FP8 throughput
RT Cores
2nd-gen, motion blur support
Minimal changes (HPC focus)
3rd-gen: Opacity Micromaps, Displaced Micro-Meshes
AI-assisted traversal, pipeline optimizations
VRAM Type
HBM2e (HPC), GDDR6X (consumer)
HBM3
GDDR6X/GDDR6
HBM3e (HPC), GDDR7 (consumer)
Max Memory Bandwidth
~1.6 TB/s (HBM2e), ~936 GB/s (GDDR6X)
Up to 3 TB/s (HBM3)
~1 TB/s (GDDR6X)
>4 TB/s (HBM3e), ~1.2 TB/s (GDDR7)
L2 Cache Size
Up to 40 MB (A100), ~6 MB (consumer)
50 MB (H100)
72 MB (RTX 4090)
100+ MB (HPC), larger per SM cache
FP64 Perf
1/2 FP32 (HPC), 1/64 FP32 (consumer)
1/2 FP32 (HPC)
Limited in pro SKUs
1/2 FP32 (HPC)
FP32 Perf
Dual datapath, +2× Turing
IPC + clock improvements
Highest clocks for gaming
Higher IPC for AI/HPC
TF32
Yes (Tensor Cores)
Yes
Yes
Yes
FP16/BF16
Yes, ~312 TFLOPS (A100)
Higher throughput, dynamic selection
Consumer AI workloads
Enhanced Transformer Engine throughput
FP8
No
Yes
Yes (pro models)
Yes, 2× Hopper throughput
FP4
No
No
No
Yes
NVLink
3.0 (600 GB/s)
4.0 (900 GB/s)
Limited/absent in consumer
5.0 (>1 TB/s)
Notable Innovations
TF32, structured sparsity
Transformer Engine, FP8
Massive L2 cache, advanced RT cores
FP4, AI-managed caching, large-model inference optimization
AMD GPU Architecture Overview and Comparison with NVIDIA
AMD approaches GPU architecture with its own design philosophy, even while addressing many of the same core challenges as NVIDIA: massive parallelism, heterogeneous compute, high memory bandwidth, and scalable interconnects.
Below, we’ll walk through
AMD’s architectural model
for the same thematic areas we covered for NVIDIA — and highlight where the two vendors align or differ.
Fundamental Architectural Components (AMD)
Compute Units (CUs)
:
AMD’s equivalent of NVIDIA’s Streaming Multiprocessors (SMs).
Each CU contains:
64 Stream Processors
(SPs, roughly equivalent to CUDA cores)
Vector ALUs
for FP32/INT32
Scalar Units
for FP32/INT32 operations shared across the CU
Matrix Cores (AI Matrix Accelerators)
in newer RDNA/CDNA architectures for mixed-precision AI workloads.
CUs are grouped into Shader Arrays and Shader Engines.
Specialized Hardware
:
Ray Accelerators
(since RDNA 2) — analogous to NVIDIA’s RT Cores.
Matrix Cores
(CDNA and RDNA 3) — compete with NVIDIA’s Tensor Cores for AI/HPC workloads.
Memory Hierarchy
:
Registers per wavefront (equivalent to warp)
Local Data Share (LDS) — similar to NVIDIA’s shared memory, software-managed.
L1 Cache and scalar caches.
L2 Cache shared across CUs.
VRAM (GDDR6, GDDR6X, or HBM depending on SKU).
Execution Paradigms (AMD)
Wavefronts
:
AMD’s equivalent to NVIDIA’s warps —
fixed at 64 threads
(vs. NVIDIA’s 32).
Like warps, wavefronts execute in lockstep using SIMD lanes.
Workgroups
:
Equivalent to NVIDIA’s CTAs (Cooperative Thread Arrays).
Workgroups contain one or more wavefronts and share LDS memory.
SIMD Execution
:
Each CU contains multiple SIMD units (typically 4×16-wide for wavefront64).
Supports predication and divergence handling similar to NVIDIA’s warp divergence model.
Scheduling
:
Hardware schedulers issue wavefront instructions to SIMD units.
AMD uses
asynchronous compute
extensively — multiple compute queues can be scheduled across the GPU concurrently.
Compute Architecture Evolution (AMD CDNA & RDNA)
CDNA (Data Center, HPC)
:
Focused on FP64, large HBM bandwidth, Infinity Fabric interconnect.
MI100, MI200, MI300 accelerators target AI training/inference and HPC workloads.
Incorporates AI Matrix Cores for FP16/BF16/FP8 acceleration.
RDNA (Gaming & Consumer)
:
RDNA 2 (2020) — introduced hardware Ray Accelerators.
RDNA 3 (2022) — chiplet-based design, higher clock speeds, improved efficiency, larger L0/L1 caches.
Trends
:
Gradual convergence with NVIDIA’s philosophy on specialized cores for AI and ray tracing.
CDNA increasingly mirrors NVIDIA’s HPC-first approach, while RDNA focuses on gaming + mixed AI features.
Floating-Point Precision Performance Evolution
AMD’s HPC GPUs (CDNA) support a similar spread of precisions:
FP64:
Full-rate or half-rate in CDNA for HPC.
FP32:
High throughput in all architectures; RDNA targets gaming efficiency.
FP16/BF16:
Supported in Matrix Cores for AI workloads.
FP8:
Introduced in MI300 for AI inference acceleration.
No FP4 yet
in shipping AMD hardware (as of 2025).
NVIDIA Comparison:
AMD lagged behind in adopting sub-16-bit formats but has now matched FP8 capabilities in data center parts.
NVIDIA maintains an edge in software-optimized mixed-precision scheduling (Transformer Engine, FP4).
Memory Architecture Evolution
VRAM Types
RDNA:
Primarily GDDR6 (with Infinity Cache to offset bandwidth needs).
CDNA:
HBM2/HBM2e (MI100), HBM3 (MI250), HBM3e (MI300).
Cache Strategies
Infinity Cache
(RDNA 2 & 3) — very large L3 cache (up to 128 MB) to minimize VRAM accesses in gaming workloads.
L1/L2 sizes generally smaller than NVIDIA’s in absolute terms, but Infinity Cache changes bandwidth behavior.
Interconnects
Infinity Fabric
— scalable, high-bandwidth interconnect across dies or GPUs.
Competes with NVIDIA NVLink; excels in multi-GPU HPC configurations.
NVIDIA Comparison:
NVIDIA tends to rely on increasing L2 cache size and raw VRAM bandwidth;
AMD focuses on large on-die caches (Infinity Cache) to improve effective bandwidth, especially in gaming SKUs.
Comparative Analysis: AMD vs. NVIDIA Architectures
Aspect
AMD Approach
NVIDIA Approach
Execution Unit
Compute Units (64 SPs)
Streaming Multiprocessors (128 CUDA cores)
Warp/Wavefront
Wavefront64
Warp32
Specialized Cores
Ray Accelerators, Matrix Cores
RT Cores, Tensor Cores
AI Acceleration
Matrix Cores (FP16/BF16/FP8)
Tensor Cores (FP16/BF16/FP8/FP4, Transformer Engine)
VRAM Strategy
Infinity Cache + VRAM
Large L2 + VRAM
HPC Interconnect
Infinity Fabric
NVLink/NVSwitch
Sub-16-bit Formats
FP8 (CDNA MI300)
FP8, FP4
Software Ecosystem
ROCm, HIP
CUDA, cuDNN, TensorRT
Consumer Focus
RDNA for gaming
Ada for gaming
HPC Focus
CDNA for data center
Hopper/Blackwell for data center
Key Takeaways
Warp/Wavefront Size Difference:
NVIDIA’s 32-thread warps offer finer granularity, while AMD’s 64-thread wavefronts can be more efficient in highly parallel workloads but risk more idle lanes under divergence.
Memory Philosophy:
NVIDIA favors enlarging L2 cache and maximizing bandwidth; AMD offsets narrower VRAM interfaces with large on-die Infinity Cache.
AI Focus:
NVIDIA’s Transformer Engine and
float4
give it a precision flexibility edge for LLM workloads, while AMD is catching up rapidly in FP8 inference.
Interconnect Strategy:
Both offer high-bandwidth interconnects (Infinity Fabric vs. NVLink), but NVIDIA currently scales to larger GPU counts in a single coherent memory space.
Software Ecosystem:
NVIDIA’s CUDA ecosystem remains more mature, but AMD’s ROCm stack has made major strides in HPC adoption.
Citation
If you found our work useful, please cite it as:
@article{Chadha2020DistilledGPUArch,
  title   = {GPU Architecture},
  author  = {Chadha, Aman},
  journal = {Distilled AI},
  year    = {2020},
  note    = {\url{https://aman.ai}}
}