import sys, os

try:
    import accelergy
    accelergy_found = True
except ImportError:
    accelergy_found = False
    pass

class Arch:
    
    def __init__(self, arch_dict, mapspace_dict, acc_paths=[]):
        self.arch_dict = arch_dict
        self.mapspace_dict = mapspace_dict

        # num of mem instances for each level
        self.mem_instances = []
        self.mac_instances = None
        # mem entries for each instance 
        self.mem_entries = [] 
        # mem blocksize for each instance
        self.mem_blocksizes = [] 

        # name to idx lookup
        self.mem_idx = {} 
        # idx to name lookup
        self.mem_name = {}
        # arch dict paths
        self.mem_paths = []

        # accelergy buffer names and components
        self.mem_acc_names = []
        self.acc_paths = acc_paths
        # access energy cost either specified in yaml file or estimated using Accelergy (in nJ)
        self.mem_access_cost = []
        self.read_bandwidth = []

        self.mac_name = "MAC"
        self.mac_acc_name = None
        self.mac_energy = 0.56*1e-3

        self.parse_arch_recursive(arch_dict['architecture']['subtree'])

        if self.mac_instances == None:
            self.mac_instances = self.mac_instances[0]
        
        self.num_mems = len(self.mem_idx.items())
        self.fanouts = self.calculate_fanouts()

        # Bypass
        self.mem_bypass_defined = None
        self.mem_stores_datatype = None
        self.mem_stores_multiple_datatypes = None
        self.mem_stored_datatypes = None
        self.parse_bypass(mapspace_dict)

        if None not in self.mem_access_cost: # energy costs in arch file
            print("Using access costs specified in architecture file.")
        elif accelergy_found: # estimate using accelergy
            print("Some buffers in arch .yaml file lack energy costs specs.")
            print("Found Accelergy installed. Estimating missing energy costs ...")
            self.gen_accelergy_costs()
        else:
            sys.exit("🔴 Could not find energy costs in arch .yaml file and Accelergy is not installed.")

        if None in self.mem_access_cost:
            sys.exit("🔴 Could not estimate all missing costs using Accelergy: {}".format(self.mem_access_cost))

        print("Bandwidths:   {}".format(' '.join([f'[L{m}] {bw}' for m, bw in enumerate(self.read_bandwidth)])))
        print("Energy costs: [MAC] {} {}\n".format(self.mac_energy, ' '.join([f'[L{m}] {en}' for m, en in enumerate(self.mem_access_cost)])))
        

    def parse_arch_recursive(self, subtree, partial_mem_path = [], partial_mem_names = []):
        partial_mem_names.append(subtree[0]['name'])
        partial_mem_path.extend(['subtree',0])

        if 'subtree' in subtree[0]:
            self.parse_arch_recursive(subtree[0]['subtree'], partial_mem_path[:], partial_mem_names[:])
        
        if 'local' in subtree[0]:
            for i, node in reversed(list(enumerate(subtree[0]['local']))):
                if node['class'] in ['DRAM','SRAM','regfile','storage']:
                    self.mem_acc_names.append(partial_mem_names + [node['name']])
                    self.mem_paths.append(partial_mem_path + ['local', i])
                    self.parse_buffer(node)
                elif node['class'] in ['compute','intmac','fpmac']:
                    self.mac_instances = node['attributes']['instances']
                    self.mac_name = node['name']
                    self.mac_acc_name = partial_mem_names + [node['name']]

    def parse_buffer(self, d):
        idx = len(self.mem_idx)
        self.mem_idx[d['name']] = idx
        self.mem_name[idx] = d['name']
        attributes = d['attributes']
        self.mem_instances.append(attributes.get('instances',1))
        self.mem_blocksizes.append(attributes.get('block-size',1))
        if 'access-cost' in attributes:
            self.mem_access_cost.append(float(attributes['access-cost'] * 1e-3))
        elif 'vector-access-energy' in attributes:
            self.mem_access_cost.append(attributes['vector-access-energy'] / self.mem_blocksizes[-1] * 1e-3)
        else:
            self.mem_access_cost.append(None)
        self.read_bandwidth.append(attributes.get('read_bandwidth',float('Inf')))

        if 'entries' in attributes:
            self.mem_entries.append(attributes['entries'])
        elif 'depth' in attributes:
            entries = attributes['depth'] * self.mem_blocksizes[-1]
            self.mem_entries.append(entries)
        else: 
            self.mem_entries.append(-1) # DRAM

    def calculate_fanouts(self) -> list:
        fanouts = []
        inner_instances = self.mac_instances
        for i in self.mem_instances:
            fanouts.append(inner_instances // i)
            inner_instances = i
        return fanouts

    def gen_accelergy_costs(self):
        sys.stdout = open(os.devnull, "w")

        from accelergy import raw_inputs_2_dicts, component_class, plug_in_path_to_obj
        from accelergy import arch_dict_2_obj, primitive_component, compound_component, ERT_generator
        
        accelergy_version = 0.3
        precision = 5
        
        raw_input_info = {'path_arglist': self.acc_paths, 'parser_version': accelergy_version}
        raw_dicts = raw_inputs_2_dicts.RawInputs2Dicts(raw_input_info)
        
        plugins = plug_in_path_to_obj.plug_in_path_to_obj(raw_dicts.get_estimation_plug_in_paths(), "lemon")

        pc_classes = {}
        cc_classes = {}

        for pc_name, pc_info in raw_dicts.get_pc_classses().items():
            pcc = component_class.ComponentClass(pc_info)
            pc_classes[pcc.get_name()] = pcc
        for cc_name, cc_info in raw_dicts.get_cc_classses().items():
            ccc = component_class.ComponentClass(cc_info)
            cc_classes[ccc.get_name()] = ccc


        for m in range(-1, self.num_mems):
            if m == -1:
                mem_name = '.'.join(self.mac_acc_name)
            else:
                if self.mem_access_cost[m] != None:
                    continue
                mem_name = '.'.join(self.mem_acc_names[m])
                    
            arch_obj = arch_dict_2_obj.arch_dict_2_obj(raw_dicts.get_flatten_arch_spec_dict(), pc_classes, cc_classes)

            primitive_components = {}
            compound_components = {}

            for arch_component in arch_obj:
                if arch_component.get_class_name() in pc_classes:
                    class_name = arch_component.get_class_name()
                    pc = primitive_component.PrimitiveComponent({'component': arch_component, 'pc_class': pc_classes[class_name]})
                    if pc.get_name() == mem_name:
                        primitive_components[mem_name] = pc
                elif arch_component.get_class_name() in cc_classes:
                    cc = compound_component.CompoundComponent({'component': arch_component, 'pc_classes': pc_classes, 'cc_classes': cc_classes})
                    if cc.get_name() == mem_name:
                        compound_components[mem_name] = cc
                else:
                    sys.exit(f'Cannot find class name {arch_component.get_class()} specified in architecture')

            ert_gen = ERT_generator.EnergyReferenceTableGenerator({'parser_version': accelergy_version,
                                                        'pcs': primitive_components, 'ccs': compound_components,
                                                        'plug_ins': plugins, 'precision': precision})

            ert = ert_gen.get_ERT()

            energy = None
            if m == -1:
                for c in ert.get_ERT_summary()['ERT_summary']['table_summary']:
                    if c['name'] == mem_name:
                        actions = c['actions']
                        for a in actions:
                            if a['name'] == 'mac_random':
                                energy = 1e-3 * float( a.get('energy', a.get('max_energy', 0) ) )

                self.mac_energy = energy
            else:
                for c in ert.get_ERT_summary()['ERT_summary']['table_summary']:
                    if c['name'] == mem_name:
                        actions = c['actions']
                        for a in actions:
                            if a['name'] in ['read','access']:
                                energy = 1e-3 * float( a.get('energy', a.get('max_energy', 0) ) ) / self.mem_blocksizes[m]

                self.mem_access_cost[m] = energy


        sys.stdout = sys.__stdout__

    def parse_bypass(self, mapspace_dict):
        self.mem_stores_datatype = []
        for _ in range(self.num_mems):
            self.mem_stores_datatype.append([True, True, True])

        self.mem_bypass_defined = [False] * self.num_mems
        self.mem_bypass_defined[-1] = True # assume DRAM always stores everything, nothing to optimize

        dt_name_idx_dict = {'Inputs': 0, 'Weights': 1, 'Outputs': 2}

        directives = mapspace_dict['mapspace']['constraints']
        for node in directives:
            for dt in node.get('bypass', []):
                m = self.mem_idx[node['target']]
                d = dt_name_idx_dict[dt]
                self.mem_stores_datatype[m][d] = False
                self.mem_bypass_defined[m] = True

        self.mem_stores_multiple_datatypes = [self.mem_stores_datatype[m].count(True) > 1 for m in range(self.num_mems)]
        self.mem_stored_datatypes = [
            [t for t,is_stored in enumerate(self.mem_stores_datatype[m]) if is_stored] 
            for m in range(self.num_mems)
        ]