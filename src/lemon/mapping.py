
class Mapping:
    
    
    def __init__(self, workload, arch, bounds, permutations, unrollings, bypass) -> None:
        self.workload = workload
        self.arch = arch
        self.bounds = bounds
        self.permutations = permutations
        self.unrollings = unrollings
        self.bypass = bypass

        self.bypass_dict = arch.mapspace_dict


    def generate_dict(self) -> dict:
        mapping = {}
        mapping['mapping'] = []

        idx_to_datatype = {0: 'Inputs', 1: 'Weights', 2: 'Outputs'}
        
        
        #for node in self.bypass_dict['mapspace']['constraints']:
        #    if node['type'] in ['datatype', 'bypass']:
        #        mapping['mapping'].append(node)

        for m, bypass in enumerate(self.bypass):
            node = {}
            node['target'] = self.arch.mem_name[m].split('[')[0]
            node['type'] = 'bypass'
            node['keep'] = []
            node['bypass'] = []
            for t in range(3):
                if bypass[t]:
                    node['keep'].append(idx_to_datatype[t])
                else:
                    node['bypass'].append(idx_to_datatype[t])
            mapping['mapping'].append(node)

        for m, (bounds, permutations) in enumerate(zip(self.bounds, self.permutations)):
            node = {}
            node['target'] = self.arch.mem_name[m].split('[')[0]
            node['type'] = 'temporal'
            node['factors'] = ' '.join(f'{dim_name}={bounds[j]}' for j, dim_name in self.workload.dim_idx_name_dict.items())
            node['permutation'] = ''.join(self.workload.dim_idx_name_dict[j] for j in permutations)
            mapping['mapping'].append(node)

        for m, unrollings in enumerate(self.unrollings):
            if self.arch.fanouts[m] == 1:
                continue
            node = {}
            node['target'] = self.arch.mem_name[m].split('[')[0]
            node['type'] = 'spatial'
            node['factors'] = ' '.join(f'{dim_name}={unrollings[j]}' for j, dim_name in self.workload.dim_idx_name_dict.items())
            node['permutation'] = 'RSPQCKN'
            mapping['mapping'].append(node)

        return mapping