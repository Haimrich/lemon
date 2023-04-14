# LEMON Artifact Evaluation Instructions

Each subdirectory in this folder contains scripts to run the experiments in the paper.

## Comparisons

All the files related to comparison with other methods are in the directories `vs_*`. These folders contains a python script to run lemon `run_lemon.py` and a script to run the other sota method. If you followed the setup steps for *lemon-only* you can only run lemon while if you setup the environment for artifact evaluation you can also get the results for the other methods. The respective reference results are located in `vs_*/results-ref` directories.

### GAMMA: Genetic Algorithm

The script for the comparison against GAMMA are provided in the `vs_gamma` directory. The results for Gamma can be obtained in the `vs_gamma/results` folder running the python script:
```shell
cd /app/evaluation/vs_gamma
python run_gamma.py
ls results/gamma
```
The script will run GAMMA for each layer of the benchmark DNNs and it will take approximately ... hours. The results for LEMON for the same target architecture and workloads can be obtained running the `run_lemon.py` script in the same folder.

The comparison plot can be obtained running the two cells in the `plots.ipynb` jupyter notebook.

### MindMappings: Gradient-Based Search

The script for the comparison against MindMappings are provided in the `vs_mindmappings` directory. First run MindMappings through the script `run_mindmappings.py`. This will produce the results for 5 layers and the corresponding workload files:
```shell
cd /app/evaluation/vs_mindmappings
python run_mindmappings.py
ls results/mindmappings
```
Then, run LEMON on the same workloads running `run_lemon.py` in the same folder. The comparison plots can be drawn using the `plots.ipynb` jupyter notebook.

### CoSA: Mixed-Integer Programming

Similarly to the other two comparisons, the scripts for comparison against CoSA are located in the `vs_cosa` folder. LEMON results can be obtained running `python run_lemon.py` while CoSA results using `run_cosa.py`. The plots can be drawn using the `plots.ipynb` jupyter notebook. 

## Other Evaluations

These experiments are carried out only using LEMON, thus they can be run with the *lemon-only* setup too.

### Optimality

Scripts to compare the mapping obtained using LEMON and the optimal mapping obtained using Timeloop exhaustive mapper can be found in the directory `vs_optimal`. The optimal mapping can be obtained running the command `bash run_timeloop.sh` while the LEMON solution can be obtained using `bash run_lemon.sh`. The `results.ipynb` notebook contains information to manually extract the EDPs and the runtime from the two obtained output files.

### Bypass Exploration

Scripts to run bypass exploration and compare the solutions with fixed bypass mappings found using LEMON are located in the folder `bypass`. The results for both optimization modes can be obtained running the script `run_lemon.py` and plots can be drawn in the `plots.ipynb` notebook.

### Semi-fixed Mapping Optimization

In `fixing` directory, running the script `run_lemon.py` and plots can be drawn in the `plots.ipynb` notebook.







