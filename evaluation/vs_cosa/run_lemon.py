import os, glob, time, yaml, json

out_dir = './results/lemon'
#arch_file = './inputs/arch/simba_v3.yaml ./inputs/arch/components'
arch_file = './inputs/arch/simba_pc_v3.yaml'
mapspace_file = './inputs/mapspace/mapspace.yaml'
lookup_file = '../workloads/lookups.yaml'

with open(lookup_file, 'r') as f:
    lookup_yaml = yaml.safe_load(f)

networks = lookup_yaml['networks']
layer_workload_lookups = lookup_yaml['layer_workload_lookups']
#networks = ['mobilenetv3']
lemon_runtimes = {}
for network in networks:
    runtime = time.time()
    output_dir = f'{out_dir}/{network}/'
    os.makedirs(output_dir, exist_ok=True)

    workload_ids = sorted(list(set(layer_workload_lookups[network])))
    workload_files = sorted(glob.glob(f'../workloads/{network}/*.yaml'))

    for w, workload_file in enumerate(workload_files):
        idx = str(w).zfill(len(str(len(workload_files))))
        cmd = f'lemon {arch_file} {mapspace_file} {workload_file} -o {output_dir}/{idx} --time_limit_single 1800'
        os.system(cmd)
        #exit()

    lemon_runtimes[network] = time.time() - runtime
    

with open(f'{out_dir}/runtime.json', 'w') as outfile:
    json.dump(lemon_runtimes, outfile)