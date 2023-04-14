import os, glob, time, yaml, json

out_dir = './results/lemon'
arch_file = './inputs/arch.yaml'
lookup_file = '../workloads/lookups.yaml'

with open(lookup_file, 'r') as f:
    lookup_yaml = yaml.safe_load(f)
    
networks = lookup_yaml['networks']
layer_workload_lookups = lookup_yaml['layer_workload_lookups']

for network in ['alexnet','vgg16']:
    for opt in [(),('fp',), ('sp',),('fp','sp')]: # flexible, fixed perm, static part, dixed perm and static part
        
        runtime = time.time()
        suffix = ''.join([f'_{o}' for o in opt])
        output_dir = f'{out_dir}{suffix}/{network}/'
        if os.path.isfile(f'{output_dir}/runtime.json'):
            continue
        os.makedirs(output_dir, exist_ok=True)

        workload_ids = sorted(list(set(layer_workload_lookups[network])))
        workload_files = sorted(glob.glob(f'../workloads/{network}/*.yaml'))

        lookup = ' '.join(map(str,layer_workload_lookups[network]))
        opts = ' '.join([f'-{o}' for o in opt])
        cmd = f'lemon {arch_file} ../workloads/{network} -lu {lookup} {opts} -o {output_dir}'
        print(cmd)
        os.system(cmd)

        runtime = {'runtime': time.time() - runtime}
        with open(f'{output_dir}/runtime.json', 'w') as outfile:
            json.dump(runtime, outfile)
        time.sleep(1)
