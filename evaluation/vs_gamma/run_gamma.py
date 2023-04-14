import os, glob, time, yaml, json

num_pops = 100
epochs = 50

out_dir = './results/gamma'
workload_template_file = './inputs/problem/problem_template.yaml'
input_config_dir = './inputs/tmp/'
lookup_file = '../workloads/lookups.yaml'

with open(lookup_file, 'r') as f:
    lookup_yaml = yaml.safe_load(f)
    
networks = lookup_yaml['networks']
layer_workload_lookups = lookup_yaml['layer_workload_lookups']

gamma_runtimes = {}
for network in networks:
    runtime = time.time()
    output_dir = f'{out_dir}/{network}/'
    os.makedirs(output_dir, exist_ok=True)

    workload_ids = sorted(list(set(layer_workload_lookups[network])))
    workload_files = sorted(glob.glob(f'../workloads/{network}/*.yaml'))

    for w, workload_file in enumerate(workload_files):

        with open(workload_file, 'r') as f:
            wd = yaml.safe_load(f)['problem']
        with open(workload_template_file, 'r') as f:
            out_workload = f.read().format(n=wd['N'], c=wd['C'], m=wd['K'], r=wd['R'], s=wd['S'], p=wd['P'], q=wd['Q'], 
                Wstride=wd['Wstride'], Hstride=wd['Hstride'], Wdilation=wd['Wdilation'], Hdilation=wd['Hdilation'])
        with open(input_config_dir + "problem.yaml", 'w') as f:
            f.write(out_workload)

        idx = str(w).zfill(len(str(len(workload_files))))
        cmd = f'python /setup/gamma-timeloop/src/main.py --fitness1 edp --config_path {input_config_dir} --num_pops {num_pops} --epochs {epochs} --report_dir {output_dir}{idx}'
        print(cmd)
        os.system(cmd)

        tl_cmd = f'timeloop-model {output_dir}{idx}/map.yaml {output_dir}{idx}/arch.yaml {output_dir}{idx}/problem.yaml -o {output_dir}{idx}'
        os.system(tl_cmd)

    gamma_runtimes[network] = time.time() - runtime
    time.sleep(2)

with open(f'{out_dir}/runtime.json', 'w') as outfile:
    json.dump(gamma_runtimes, outfile)