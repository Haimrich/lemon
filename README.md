# 🍋 LEMON
This repository contains the reference implementation of LEMON mapper and the scripts to run the experiments in the paper ***Memory-Aware DNN Algorithm-Hardware Mapping via Integer Linear Programming*** presented at the 20th ACM International Conference on Computing Frontiers.

## Setup

- Clone the repository
- Acquire a [free academic Gurobi Web License](https://www.gurobi.com/features/academic-wls-license/)

Now you can decide to install LEMON only or setup the environment to replicate the comparisons against the state-on-the-art presented in the paper.

### A. LEMON Only

If you want to install LEMON only you can follow one of the following methods:

#### Option 1: python package

- [Install the Gurobi license](https://www.gurobi.com/documentation/10.0/quickstart_mac/web_license_service_wls_cl.html#subsection:clientlicenseWLS) in your system
- Install [Timeloop Model](https://timeloop.csail.mit.edu/timeloop/installation) and [Accelergy](https://timeloop.csail.mit.edu/accelergy/installation) for mapping evaluation.
- Install the LEMON package
```shell
cd lemon
pip install -e .
```

#### Option 2: docker
- Put the downloaded `license.lic` file in `lemon/docker/put-gurobi-license-here`
- Build and run the docker image
```shell
cd lemon/docker
docker-compose -f docker-compose.lemon.yaml run lemon
```
#### Option 3: devcontainer
- Put the downloaded `license.lic` file in `lemon/docker/put-gurobi-license-here`
- Open the `lemon` folder in [VSCode](https://code.visualstudio.com/)
- Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VSCode
- Open the Command Palette (`F1`), select "Reopen Folder in Container" and select **lemon-only**

More information about devcontainers in VSCode can be found [here](https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container).

### B. Artifact Evaluation

#### nvidia-docker

If you want to run the artifact evaluation you can use the provided docker image.  One of the other state-of-the-art mapper is implemented using PyTorch, for this reason a CUDA GPU is required. To make the GPU accessable in a docker container, `nvidia-docker2` has to be installed in the host system following the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

#### Gurobi license

Put the downloaded Gurobi `license.lic` file in the `docker/put-gurobi-license-here` folder.

#### Build the image and run the container

Now it is possible to build the docker image and launch the container either using the following commands:

```Shell
cd docker
docker-compose -f docker-compose.ae.yaml run lemon-ae
```

or, if you are using VSCode, you can use the provided devcontainer as follows:
- Open the `lemon` folder in VSCode
- Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VSCode
- Open the Command Palette (`F1`), select "Reopen Folder in Container" and select **lemon-artifact-evaluation**

More information about devcontainers in VSCode can be found [here](https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container).

#### Run the experiments

See [evaluation](evaluation/README.md).

## Usage

For information about the `lemon` command arguments type:
```shell
lemon --help
```

```

oooo                                                    
`888                                                    
 888   .ooooo.  ooo. .oo.  .oo.    .ooooo.  ooo. .oo.   
 888  d88' `88b `888P"Y88bP"Y88b  d88' `88b `888P"Y88b  
 888  888ooo888  888   888   888  888   888  888   888  
 888  888    .o  888   888   888  888   888  888   888  
o888o `Y8bod8P' o888o o888o o888o `Y8bod8P' o888o o888o         
                                                        
usage: lemon [-h] [-o OUTPUT] [-lu [LAYER_LOOKUP ...]] [-ew ENERGY_WEIGHT] [-lw LATENCY_WEIGHT] [-ot] [-ob] [-fb] [-sp] [-fp] [-tls TIME_LIMIT_SINGLE]
             inputs [inputs ...]

positional arguments:
  inputs                Input YAML files: architecture, mapspace and workloads required

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output Directory. Default = ./output
  -lu [LAYER_LOOKUP ...], --layer_lookup [LAYER_LOOKUP ...]
                        Workload ID for each layer. Default = 0 1 2 ... W-1
  -ew ENERGY_WEIGHT, --energy_weight ENERGY_WEIGHT
                        Energy weight in the objective. Default = 1
  -lw LATENCY_WEIGHT, --latency_weight LATENCY_WEIGHT
                        Latency weight in the objective. Default = 10
  -ot , --objective_type 
                        MIP model objective type: 'blended' for weighted sum of energy and latency as obj (fastest), 'latency-energy' to first minimize
                        latency and then energy, 'energy-latency' for minimizing energy first and then latency, 'quadratic' to minimize EDP (this
                        transforms the model in a MIQP, slowest). Default = blended
  -ob, --optimize_bypass
                        Optimize datatype memory bypass not specified in mapspace file (transforms model in MICQP, slower)
  -fb, --fix_bypass     Make sure to have same bypass directives for all layers
  -sp, --static_partitioning
                        Make sure buffer partitioning is the same for all the layers
  -fp, --fix_permutations
                        Make sure temporal permutations are the same for all the layers
  -tls TIME_LIMIT_SINGLE, --time_limit_single TIME_LIMIT_SINGLE
                        Search time limt (seconds) when mapping a single layer
```

### End-to-end Mode

LEMON can map all the layers of a DNN model simultaneously. In this case a layer-workload-lookup list has to be provided using the option `-lu`. Each integer in the provided list represent the workload identifier of a layer. The workload files has to be alphabetically ordered in a folder provided as input. See the `evaluation/fixing` folder for end-to-end mode usage examples.

### Bypass Optimization Mode

The bypass optimizaion mode can be enable using the option `-ob`. This transforms the problem in a MICQP and could require more time to solve. See the `evaluation/bypass` folder examples.

### Objective Types

LEMON supports four optimization objetives that can be configured using the `-ot` option:
- `blended`: this is the default one and consists in the linear combination of energy and latency. The weights can be adjusted using the `-ew` and `-lw` options respectively.
- `latency-energy`: first minimize latency and then minimize energy
- `energy-latency`: first minimize energy and then minimize latency
- `quadratic`: minimize the EDP, i.e. the product of latency and energy. This transforms the problem in MIQP and could require more time to solve.

## Citation

If you find this repository useful please cite:
```BibTeX
@inproceedings{russo2023lemon,
  title={Memory-Aware DNN Algorithm-Hardware Mapping via Integer Linear Programming},
  author={Russo, Enrico and Palesi, Maurizio and Ascia, Giuseppe and Patti, Davide and Monteleone, Salvatore and Catania, Vincenzo},
  booktitle={Proceedings of the 20th ACM International Conference on Computing Frontiers},
  year={2023},
  doi={10.1145/3587135.3592206},
}
```

> Enrico Russo, Maurizio Palesi, Giuseppe Ascia, Davide Patti, Salvatore Monteleone, and Vincenzo Catania. 2023. ***Memory-Aware DNN Algorithm-Hardware Mapping via Integer Linear Programming.*** In 20th ACM International Conference on Computing Frontiers (CF ’23), May 9–11, 2023, Bologna, Italy. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3587135.3592206
