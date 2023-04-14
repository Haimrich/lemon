import math
import shutil
import sys
import os
import subprocess
import re
import logging

import yaml

def run_timeloop(acc_paths, workload_path, map, output_dir):
    if shutil.which("timeloop-model") == None:
        sys.exit("🔴 Error: cant find timeloop-model executable in PATH")

    os.makedirs(output_dir, exist_ok=True)
    out_mapping_file = output_dir + "/🍋.yaml"

    with open(out_mapping_file, 'w') as f:
        yaml.dump(map.generate_dict(), f)

    cmd = f"timeloop-model {' '.join(acc_paths)} {workload_path} {out_mapping_file} -o {output_dir}"
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    out_stats_file = output_dir + "/timeloop-model.stats.txt"
    if not os.path.isfile(out_stats_file):
        print(proc.stdout)
        sys.exit("🔴 Error: timeloop evaluation failed")

    utilization, efficiency, cycles, energy = extract_tl_stats(out_stats_file)
    print(f"Utilization = {utilization} | pJ/Compute = {efficiency} | Cycles: {cycles} | Energy: {energy} uJ | EDP: {energy*cycles}")
        

def extract_tl_stats(file_path):
  with open(file_path, "r") as f:
    stats = f.read()

    res = re.findall(r'Utilization:\s(\d+.\d+)\s', stats, re.DOTALL)
    utilization = float(res[0])

    res = re.findall(r'Total\s+= (\d+.\d+)\s', stats, re.DOTALL)
    efficiency = float(res[0])

    res = re.findall(r'Cycles:\s(\d+)', stats, re.DOTALL)
    cycles = int(res[0])

    res = re.findall(r'Energy:\s(\d+.\d+)\suJ', stats, re.DOTALL)
    energy = float(res[0])

    f.close()

  return utilization, efficiency, cycles, energy


def get_divisors_gen(n):
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(int(n / i))
    for divisor in reversed(large_divisors):
        yield divisor


def nested_dict_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic[key]
    dic[keys[-1]] = value


class GurobiFilter(logging.Filter):
    def __init__(self, name="GurobiFilter"):
        super().__init__(name)

    def filter(self, record):
        return False


def suppress_gurobi_logger():
    grbfilter = GurobiFilter()
    grblogger = logging.getLogger('gurobipy')
    if grblogger is not None:
        grblogger.addFilter(grbfilter)
        grblogger = grblogger.getChild('gurobipy')
        if grblogger is not None:
            grblogger.addFilter(grbfilter)


