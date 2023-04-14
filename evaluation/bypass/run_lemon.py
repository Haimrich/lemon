import os, glob, time, yaml, json

out_dir = './results/lemon'
arch_file = './inputs/arch.yaml'
lookup_file = '../workloads/lookups.yaml'

with open(lookup_file, 'r') as f:
    lookup_yaml = yaml.safe_load(f)
    
networks = lookup_yaml['networks']
layer_workload_lookups = lookup_yaml['layer_workload_lookups']

for network in networks:
    for opt in range(2): 
        runtime = time.time()
        
        suffix = '' if opt == 0 else '_ob'
        output_dir = f'{out_dir}{suffix}/{network}/'
        os.makedirs(output_dir, exist_ok=True)

        workload_ids = sorted(list(set(layer_workload_lookups[network])))
        workload_files = sorted(glob.glob(f'../workloads/{network}/*.yaml'))

        for w, workload_file in enumerate(workload_files):
            print(f'[RUN] Network: {network} - Workload {w}')
            idx = str(w).zfill(len(str(len(workload_files))))

            opts = '-ob' if opt == 1 else ''
            cmd = f'lemon {arch_file} {workload_file} {opts} -o {output_dir}/{idx}'
            os.system(cmd)
            
        runtime = {'runtime': time.time() - runtime}
        with open(f'{output_dir}/runtime.json', 'w') as outfile:
            json.dump(runtime, outfile)
        time.sleep(1)
