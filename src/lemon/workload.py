import numpy as np

from lemon.utils import get_divisors_gen

class Workload:

    def __init__(self, path, workload_dict):
        self.path = path
        
        self.prob_dict = workload_dict['problem'] \
            if isinstance(workload_dict['problem']['shape'], str) else workload_dict['problem']['instance']

        self.num_dims = 7
        self.dim_idxs = list(range(self.num_dims))
        self.dim_idx_name_dict = {0: 'R', 1: 'S', 2: 'P', 3: 'Q', 4: 'C', 5: 'K', 6: 'N'}
        self.dim_name_idx_dict = {v: k for k, v in self.dim_idx_name_dict.items()}

        self.bounds = [1] * len(self.dim_name_idx_dict)

        for key, value in self.prob_dict.items():
            if ('stride' in key or 'dilation' in key or key == 'shape'):
                continue    
            dim_idx = self.dim_name_idx_dict[key]
            self.bounds[dim_idx] = value
        
        self.stride = (self.prob_dict.get('Wstride',1), self.prob_dict.get('Hstride',1))
        self.dilation = (self.prob_dict.get('Wdilation',1), self.prob_dict.get('Hdilation',1))
        self.macs = np.prod(self.bounds)

        self.divisors = []
        for j, dim in enumerate(self.bounds):
            divs = sorted(list(get_divisors_gen(dim)))
            self.divisors.append(divs)
        
        self.weight = 1

        self.O = [ # Dim-Datatype relevancy matrix
        #t: 0: Inputs,   1: Weights,  2: Outputs    j:
            [1,          1,           0],         # 0: R
            [1,          1,           0],         # 1: S
            [1,          0,           1],         # 2: P
            [1,          0,           1],         # 3: Q
            [1,          1,           0],         # 4: C
            [0,          1,           1],         # 5: K
            [1,          0,           1],         # 6: N
        ]