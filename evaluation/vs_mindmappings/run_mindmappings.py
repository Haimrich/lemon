import os, glob, time, yaml, json, shutil

out_dir = './results/mindmappings'

workloads =  [ #N C K R S P Q
    [16, 64,128,3,3,112,112], # VGG_conv2_1
    [32, 64,192,3,3,56,56], # Inception_conv2_1x1
    [8, 96,256,5,5,27,27],  # AlexNet_conv2
    [8, 384,384,3,3,13,13], # AlexNet_conv4
    [16,128,128,3,3,28,28], # ResNet_Conv3_0_3x3
    [16,256,256,3,3,14,14] # ResNet_Conv3_0_3x3
]

mindmappings_runtimes = {}
for i, workload in enumerate(workloads):
    runtime = time.time()
    output_dir = f'{out_dir}/{i}/'
    #os.makedirs(output_dir, exist_ok=True)

    problem = ' '.join(map(str, workload))
        
    cmd = f'python3 /setup/mindmappings/mindmappings/optimize.py --command search --algorithm CNN-layer --problem {problem} --maxsteps 1000'
    print(cmd)
    os.system(cmd)

    src = './tmp/timeloop/outputs_CNN-layer/grun_0/'
    dst = f'./{output_dir}'
    shutil.move(src, dst)

    mindmappings_runtimes[i] = time.time() - runtime

with open(f'{out_dir}/runtime.json', 'w') as outfile:
    json.dump(mindmappings_runtimes, outfile)