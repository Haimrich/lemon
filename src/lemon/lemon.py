import sys, argparse, pathlib

import yaml

from lemon.arch import Arch
from lemon.workload import Workload
from lemon.model import run_optimization
from lemon.utils import run_timeloop

def main():
    banner = """
oooo                                                    
`888                                                    
 888   .ooooo.  ooo. .oo.  .oo.    .ooooo.  ooo. .oo.   
 888  d88' `88b `888P"Y88bP"Y88b  d88' `88b `888P"Y88b  
 888  888ooo888  888   888   888  888   888  888   888  
 888  888    .o  888   888   888  888   888  888   888  
o888o `Y8bod8P' o888o o888o o888o `Y8bod8P' o888o o888o         
                                                        """
    print(banner)
    parser = argparser()
    args = parser.parse_args()

    if args.fix_bypass and not args.optimize_bypass:
        parser.error("--fix_bypass requires --optimize_bypass.")

    input_files = args.inputs
    layer_workload_lookup = args.layer_lookup
    output_dir = args.output

    yaml_files = []
    for path_str in input_files:
        path = pathlib.Path(path_str).resolve()
        if path.exists():
            if path.is_file():
                yaml_files.append(path)
            else:
                yaml_files_nest = sorted(path.glob("**/*.yaml"))
                yaml_files.extend(yaml_files_nest)
        else:
            sys.exit(f"🔴 Error: '{path}' not found")

    workloads = []
    arch_dict = None
    mapspace_dict = {'mapspace': {'constraints': []}}
    acc_paths = []

    for yf in yaml_files:
        with open(yf, 'r') as f:
            content = yaml.safe_load(f)
            if 'problem' in content:
                workloads.append(Workload(yf, content))
            if 'architecture' in content:
                if arch_dict != None:
                    sys.exit("🔴 Error: more than one input architecture provided.")
                arch_dict = content
                acc_paths.append(str(yf))
            if 'mapspace' in content:
                mapspace_dict = content
            if 'compound_components' in content:
                acc_paths.append(str(yf))
    
    # Check for missing files and errors in lookup
    if arch_dict == None:
        sys.exit("🔴 Error: missing input architecture file.")
    if len(workloads) == 0:
        sys.exit("🔴 Error: missing input workload file.")
    if len(set(layer_workload_lookup)) > len(workloads) and len(layer_workload_lookup) != 0:
        sys.exit("🔴 Error: malformed layer-workload lookup.")

    # Set weight for each workload
    for l in set(layer_workload_lookup):
        workloads[l].weight = layer_workload_lookup.count(l)

    # Inizialize Arch object
    arch = Arch(arch_dict, mapspace_dict, acc_paths)

    # Run optimization
    mappings, runtime = run_optimization(
        arch, workloads, 
        args.energy_weight, args.latency_weight, args.objective_type,
        args.optimize_bypass, args.fix_bypass,
        args.static_partitioning,
        args.fix_permutations,
        args.time_limit_single
        )

    print("\n⌚ Evaluating solutions using Timeloop.\n")
    # Evaluate using Timeloop
    for w, map in enumerate(mappings):
        print(f"   Workload {w}: ", end='')
        w_output_dir = output_dir + "/" + str(w).zfill(len(str(len(mappings))))
        if len(mappings) == 1:
            w_output_dir = output_dir
        run_timeloop(acc_paths, workloads[w].path, map, w_output_dir)

    print("\n🍋 Done.\n")


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs="+", type=str,
                        help='Input YAML files: architecture, mapspace and workloads required')
    parser.add_argument('-o', '--output', type=str, default='./output',
                        help='Output Directory. Default = ./output')
    parser.add_argument('-lu', '--layer_lookup', nargs="*", type=int, default=[],
                        help='Workload ID for each layer. Default = 0 1 2 ... W-1')
    parser.add_argument('-ew', '--energy_weight', type=float, default=1,
                        help='Energy weight in the objective. Default = 1')
    parser.add_argument('-lw', '--latency_weight', type=float, default=10,
                        help='Latency weight in the objective. Default = 10')
    parser.add_argument('-ot', '--objective_type', choices=['blended', 'latency-energy', 'energy-latency', 'quadratic'], default='blended',
                        help="MIP model objective type: 'blended' for weighted sum of energy and latency as obj (fastest), " \
                        "'latency-energy' to first minimize latency and then energy, 'energy-latency' for minimizing energy first and then latency, "\
                        "'quadratic' to minimize EDP (this transforms the model in a MIQP, slowest). Default = blended",
                        metavar='')
    parser.add_argument('-ob', '--optimize_bypass', default=False, action='store_true',
                        help='Optimize datatype memory bypass not specified in mapspace file (transforms model in MICQP, slower)')
    parser.add_argument('-fb', '--fix_bypass', default=False, action='store_true',
                        help='Make sure to have same bypass directives for all layers')
    parser.add_argument('-sp', '--static_partitioning', default=False, action='store_true',
                        help='Make sure buffer partitioning is the same for all the layers')
    parser.add_argument('-fp', '--fix_permutations', default=False, action='store_true',
                        help='Make sure temporal permutations are the same for all the layers')
    parser.add_argument('-tls', '--time_limit_single', type=int, default=200,
                        help='Search time limt (seconds) when mapping a single layer')
    return parser


if __name__ == "__main__":
    main()