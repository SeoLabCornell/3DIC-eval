# 3D IC Architecture Evaluation and Optimization with Digital Compute-in-Memory Designs (ISLPED, 2024)

This is the artifact of the paper '**3D IC Architecture Evaluation and Optimization with Digital Compute-in-Memory Designs**' presented in *ACM/IEEE International Symposium on Low Power Electronics and Design (ISLPED), 2024*.

## Updates

- **[Aug. 02, 2024]:** Initial version of the artifact is published together with the camera-ready version of our **ISLPED paper**.

## How to use?
### e_breakdown.py
Generates energy breakdown of hardware given PE array size, buffer size and workload.
- Usage: <span style="font-family: Consolas"> python e_breakdown.py -x xdim -y ydim --buf bufsize --wk workload

### cycle_util.py
Generates cycles and utilization of hardware given PE array size, buffer size and workload.
- Usage: <span style="font-family: Consolas"> python cycle_util.py -x xdim -y ydim --buf bufsize --wk workload

### energy_sweep.py
Generates energy breakdown across 8 different AI models and different sizes of PE array and buffer.
- Usage: <span style="font-family: Consolas"> python energy_sweep.py


## Authors
[Hyung Joon Byun](https://sites.google.com/view/hjbyun), [Udit Gupta](https://ugupta.com/), and [Jae-sun Seo](https://seo.ece.cornell.edu/). School of ECE, Cornell Tech.

## Cite Us
**Publication:** *3D IC Architecture Evaluation and Optimization with Digital Compute-in-Memory Designs* (Byun et al., ISLPED, 2024).

### Acknowledgement

This work was supported in part by the Center for the Co-Design of Cognitive Systems (CoCoSys) in JUMP 2.0, a Semiconductor Research Corporation (SRC) Program sponsored by the Defense Advanced Research Projects Agency (DARPA).

