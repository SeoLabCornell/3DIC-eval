# 3D IC Architecture Evaluation and Optimization with Digital Compute-in-Memory Designs (ISLPED, 2024)

This is the artifact of the paper '**3D IC Architecture Evaluation and Optimization with Digital Compute-in-Memory Designs**' presented in *ACM/IEEE International Symposium on Low Power Electronics and Design (ISLPED), 2024*.

## Updates

- **[Aug. 06, 2024]:** Initial version of the artifact is published together with the camera-ready version of our **ISLPED paper**.

## How to use?
- To get the argument information, type <code> python FILENAME.py -h</code> for the following python scripts.
- In the <code>workload</code> folder, 8 different AI workloads are loaded as csv files. You can add your own model dimensions following the other file formats. Refer to <code>workload.py</code> which reads and makes class from the csv file. 
### e_breakdown.py
Generates energy breakdown of hardware given PE array size, buffer size and workload.
- Usage: <code>python e_breakdown.py -x xdim -y ydim --buf bufsize --wk workload</code>

### cycle_util.py
Generates cycles and utilization of hardware given PE array size, buffer size and workload.
- Usage: <code>python cycle_util.py -x xdim -y ydim --buf bufsize --wk workload</code>

### energy_sweep.py
Generates energy breakdown across 8 different AI models and different sizes of PE array and buffer.
- Usage: <code>python energy_sweep.py</code>
- If you change the number of workloads, then the output figure might crash. In that case, change the dimension of the subplots at the end of the script.


## Authors
[Hyung Joon Byun](https://sites.google.com/view/hjbyun), [Udit Gupta](https://ugupta.com/), and [Jae-sun Seo](https://seo.ece.cornell.edu/). School of ECE, Cornell Tech.

## Cite Us
**Publication:** *3D IC Architecture Evaluation and Optimization with Digital Compute-in-Memory Designs* (Byun et al., ISLPED, 2024).

### Acknowledgement

This work was supported in part by the Center for the Co-Design of Cognitive Systems (CoCoSys) in JUMP 2.0, a Semiconductor Research Corporation (SRC) Program sponsored by the Defense Advanced Research Projects Agency (DARPA).

