import os, glob, time, yaml, json

out_dir = './results/lemon'
arch_file = './inputs/arch/arch.yaml'
mapspace_file = './inputs/mapspace/mapspace.yaml'
lookup_file = '../workloads/lookups.yaml'

workload_files = sorted(glob.glob('./results/mindmappings/*/prob.yaml'))

lemon_runtimes = {}
for i, workload_file in enumerate(workload_files):
    runtime = time.time()
    output_dir = f'{out_dir}/{i}'
    os.makedirs(output_dir, exist_ok=True)

    cmd = f'lemon {arch_file} {mapspace_file} {workload_file} -o {output_dir}'
    os.system(cmd)
    #exit()

    lemon_runtimes[i] = time.time() - runtime
    

with open(f'{out_dir}/runtime.json', 'w') as outfile:
    json.dump(lemon_runtimes, outfile)