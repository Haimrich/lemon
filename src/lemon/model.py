import time, sys

import numpy as np
import gurobipy as gp

from lemon.utils import suppress_gurobi_logger
from lemon.mapping import Mapping


def run_optimization(
    arch, workloads, 
    energy_weight, latency_weight, objective_type,
    optimize_bypass, fix_bypass,
    static_partitioning,
    fix_permutations,
    time_limit_single
    ):
    suppress_gurobi_logger() # avoids double print of gurobi log

    model_build_time = time.time()
    print(f"⚙️  Building MIP model...\n")

    model = gp.Model(name="Lemon")

    num_workloads = len(workloads)
    num_datatypes = 3
    num_mems = arch.num_mems

    MAX_BOUND = 1e4 # keeping right side of inequalities in the order of 1e4

    max_macs = max(w.macs for w in workloads)
    max_macs_scale_factor = MAX_BOUND / (1.02 * max_macs)
    max_macs_scale_factor_log = np.log(max_macs_scale_factor)

    macs_scale_factors = [MAX_BOUND / (1.02 * w.macs) for w in workloads]
    macs_scale_factor_logs = np.log(macs_scale_factors)
    
    max_entries = max(arch.mem_entries[m] for m in range(num_mems))
    max_util_scale_factor = MAX_BOUND / (1.02 * max_entries)
    max_util_scale_factor_log = np.log(max_util_scale_factor)

    util_scale_factors = [MAX_BOUND / (1.02 * arch.mem_entries[m]) for m in range(num_mems-1)]
    util_scale_factor_logs = np.log(util_scale_factors)
    
    model.setParam("FuncMaxVal", MAX_BOUND)
    model.setParam("ScaleFlag", 2)
    model.setParam("Presolve", 2)
    model.setParam("Heuristics", 0.3)
    #model.setParam("NumericFocus", 2)
    #model.setParam('MIPGap', 0.025)
    model.setParam("FeasibilityTol", 3e-3)

    model.setParam("NodefileStart", 0.5)
    model.setParam("NodefileDir", '.')

    #model.setParam('Cuts', 2)

    # Decision Variables: loop Bounds

    xb = gp.tupledict()
    for w, workload in enumerate(workloads):
        for m in range(num_mems):
            for s in range(2):
                for j, divs in enumerate(workload.divisors):
                    for i, div in enumerate(divs):
                        vname = f"XB_({w},{m},{s},{j},{i})"
                        xb[w,m,s,j,i] = model.addVar(vtype=gp.GRB.BINARY, name=vname)
                    
    # Decision Variables: temporal loop permutations
    
    if not fix_permutations or True:
        xp = gp.tupledict()
        for w, workload in enumerate(workloads):
            for m in range(num_mems):
                for p, _ in enumerate(workload.bounds):
                    for j, dim in enumerate(workload.bounds):
                        vname = f"XP_({w},{m},{p},{j})"
                        xp[w,m,p,j] = model.addVar(vtype=gp.GRB.BINARY, name=vname)


    # Decision Variables: memory datatype bypass (1 if stored)

    xd = gp.tupledict()
    if fix_bypass: # same for all w
        for m in range(num_mems):
            for t in range(num_datatypes):
                vname = f"XD_f_({m},{t})"
                xd_mt = model.addVar(vtype=gp.GRB.BINARY, name=vname)
                for w, workload in enumerate(workloads):
                    xd[w,m,t] = xd_mt
    else:
        for w, workload in enumerate(workloads):
            for m in range(num_mems):
                for t in range(num_datatypes):
                    vname = f"XD_({w},{m},{t})"
                    xd[w,m,t] = model.addVar(vtype=gp.GRB.BINARY, name=vname)
    
    # Constraint: product of loop dim equal to prob dim -> sum of log equal to log of dim

    for w, workload in enumerate(workloads):
        for j, divs in enumerate(workload.divisors):
            dim = workload.bounds[j]
            log_bound_sum = 0
            for m in range(num_mems): 
                for s in range(2):
                    for i, div in enumerate(divs):
                        log_bound_sum += xb[w,m,s,j,i] * np.log(div)
            model.addConstr(log_bound_sum == np.log(dim), f"C_dim_factorization_({w},{j})")
    
    # Constraint: only one factor per loop

    for w, workload in enumerate(workloads):
        for m in range(num_mems): 
            for s in range(2):
                for j, divs in enumerate(workload.divisors):
                    loop_sum = xb.sum(w,m,s,j,'*')
                    model.addConstr(loop_sum == 1, f"C_one_bound_per_loop_({w},{m},{s},{j})")
        
    # Constraint: one loop per perm level and one perm level per loop

    if not fix_permutations or True:
        for w, workload in enumerate(workloads):
            for m in range(num_mems):
                for j, divs in enumerate(workload.divisors):
                    loop_sum_b = 1 - xb[w,m,1,j,0]
                    loop_sum_p = xp.sum(w,m,'*',j)
                    model.addConstr(loop_sum_p == loop_sum_b, f"C_one_perm_level_per_loop_if_bound_neq_1_({w},{m},{j})")
                for p, _ in enumerate(workload.divisors):
                    perm_level_sum = xp.sum(w,m,p,'*')
                    model.addConstr(perm_level_sum <= 1, f"C_max_one_loop_per_perm_level_({w},{m},{p})")

    # Constraint: spatial fanout has to be compatible with available instances in architecture

    for w, workload in enumerate(workloads):
        for m in range(num_mems):
            spatial_fanout_log = 0
            for j, divs in enumerate(workload.divisors):
                for i, div in enumerate(divs):
                    spatial_fanout_log += xb[w,m,0,j,i] * np.log(div)
            model.addConstr(spatial_fanout_log <= np.log(arch.fanouts[m]+0.1), f"C_spatial_fanout_({w},{m})")

    # Constraint: constraint bypass based on user provided mapspace and options

    for w, workload in enumerate(workloads):
        for m in range(num_mems):
            for t in range(num_datatypes):
                if arch.mem_bypass_defined[m] or not optimize_bypass:
                    model.addConstr(xd[w,m,t] == int(arch.mem_stores_datatype[m][t]))

    # Constraint: fix permutations if asked

    if fix_permutations:
        gxp = gp.tupledict()
        for m in range(num_mems):
            for p, _ in enumerate(workload.bounds):
                for j, dim in enumerate(workload.bounds):
                    vname = f"GXP_({m},{p},{j})"
                    gxp[m,p,j] = model.addVar(vtype=gp.GRB.BINARY, name=vname)

        for m in range(num_mems):
            for j, divs in enumerate(workload.divisors):
                model.addConstr(gxp.sum(m,'*',j) <= 1, f"C_g_one_perm_level_per_loop_({m},{j})")
            for p, _ in enumerate(workload.divisors):
                model.addConstr(gxp.sum(m,p,'*') <= 1, f"C_g_max_one_loop_per_perm_level_({m},{p})")

        #for w, workload in enumerate(workloads):
        #    for m in range(num_mems):
        #        for p, _ in enumerate(workload.bounds):
        #            for j, dim in enumerate(workload.bounds):
        #                model.addConstr(xp[w,m,p,j] <= gxp[m,p,j], name=f'C_fix_perm_({w},{m},{p},{j})')  

        
        for m in range(num_mems):
            for p, _ in enumerate(workload.bounds):
                for j, dim in enumerate(workload.bounds):
                    model.addGenConstrOr(gxp[m,p,j], xp.select('*',m,p,j), name=f'C_fix_perm_({m},{p},{j})')
    old_fp = fix_permutations
    fix_permutations = False

    # Expression: calculating buffer utilizations

    buf_util_log = {}
    buf_util = {}
    yu = {}

    for w in range(num_workloads):
        for m in range(num_mems):
            for t in range(num_datatypes):
                buf_util_log[w,m,t] = 0
                buf_util[w,m,t] = 0

    for w, workload in enumerate(workloads):
        factors = {}

        for m in range(num_mems-1):
            optimize_bypass_m = optimize_bypass and not arch.mem_bypass_defined[m]

            # Product of loop bounds for each problem dimension up to the each memory level (R_m, S_m, P_m, ecc.)
            for j, divs in enumerate(workload.divisors):
                factors[m,j] = 0
                for m_ in range(m+1):
                    for s in range(2):
                        for i, div in enumerate(divs):
                            factors[m,j] += xb[w,m_,s,j,i] * np.log(div)

            # Inputs
            t = 0 
            if arch.mem_stores_datatype[m][t] or optimize_bypass_m: 

                # width
                wc = 1 - workload.stride[0] - workload.dilation[0]
                xw, xwr, xwp, yw, w_log = [], [], [], [], 0
                for dp, p_ in enumerate(workload.divisors[2]):
                    for dr, r_ in enumerate(workload.divisors[0]):
                        xwr.append(np.log(r_))
                        xwp.append(np.log(p_))
                        xw.append( model.addVar(vtype=gp.GRB.BINARY, name=f"XW({w},{m},{dp},{dr})") )
                        yw.append(workload.stride[0] * p_ + workload.dilation[0] * r_ + wc)
                        w_log += xw[-1] * np.log(yw[-1])
                model.addConstr(gp.quicksum(xw) == 1, name=f"C_xw_({w},{m})")
                model.addConstr(gp.LinExpr(xwr, xw) == factors[m,0], name=f"C_xwr_({w},{m})")
                model.addConstr(gp.LinExpr(xwp, xw) == factors[m,2], name=f"C_xwp_({w},{m})")

                # height
                hc = 1 - workload.stride[1] - workload.dilation[1]
                xh, xhs, xhq, yh, h_log = [], [], [], [], 0
                for dq, q_ in enumerate(workload.divisors[3]):
                    for ds, s_ in enumerate(workload.divisors[1]):
                        xhs.append(np.log(s_))
                        xhq.append(np.log(q_))
                        xh.append( model.addVar(vtype=gp.GRB.BINARY, name=f"XH({w},{m},{dq},{ds})") )
                        yh.append(workload.stride[1] * q_ + workload.dilation[1] * s_ + hc)
                        h_log += xh[-1] * np.log(yh[-1])
                model.addConstr(gp.quicksum(xh) == 1, name=f"C_xh_({w},{m})")
                model.addConstr(gp.LinExpr(xhs, xh) == factors[m,1], name=f"C_xhs_({w},{m})")
                model.addConstr(gp.LinExpr(xhq, xh) == factors[m,3], name=f"C_xhq_({w},{m})")
                
                # util log
                buf_util_log[w,m,t] = (w_log + h_log + factors[m,4] + factors[m,6])

                # util
                if arch.mem_stores_multiple_datatypes[m] or optimize_bypass_m:
                    xu, xuw, xuh, xuc, xun = [], 0, 0, 0, 0
                    yu[m,t] = []
                    for dw, w_ in enumerate(yw):
                        for dh, h_ in enumerate(yh):
                            for dc, c_ in enumerate(workload.divisors[4]):
                                for dn, n_ in enumerate(workload.divisors[6]):
                                    val = w_*h_*c_*n_
                                    if val <= arch.mem_entries[m]:
                                        xu.append( model.addVar(vtype=gp.GRB.BINARY, name=f"XU({w},{m},{t},{dw},{dh},{dc},{dn})") )
                                        xuw += xu[-1] * np.log(w_)
                                        xuh += xu[-1] * np.log(h_)
                                        xuc += xu[-1] * np.log(c_)
                                        xun += xu[-1] * np.log(n_)
                                        yu[m,t].append(val)
                                        buf_util[w,m,t] += xu[-1] * yu[m,t][-1]
                    model.addConstr(gp.quicksum(xu) == 1, name=f"C_xu_({w},{m},{t})")
                    model.addConstr(xuw == w_log, name=f"C_xuw_({w},{m},{t})")
                    model.addConstr(xuh == h_log, name=f"C_xuh_({w},{m},{t})")
                    model.addConstr(xuc == factors[m,4], name=f"C_xuc_({w},{m},{t})")
                    model.addConstr(xun == factors[m,6], name=f"C_xun_({w},{m},{t})")

            # Weights

            t = 1
            if arch.mem_stores_datatype[m][t] or optimize_bypass_m:
                buf_util_log[w,m,t] = factors[m,0] + factors[m,1] + factors[m,4] + factors[m,5]

                if arch.mem_stores_multiple_datatypes[m] or optimize_bypass_m:
                    xu, xur, xus, xuc, xuk = [], 0, 0, 0, 0
                    yu[m,t] = []
                    for dr, r_ in enumerate(workload.divisors[0]):
                        for ds, s_ in enumerate(workload.divisors[1]):
                            for dc, c_ in enumerate(workload.divisors[4]):
                                for dk, k_ in enumerate(workload.divisors[5]):
                                    val = r_*s_*c_*k_
                                    if val <= arch.mem_entries[m]:
                                        xu.append( model.addVar(vtype=gp.GRB.BINARY, name=f"XU({w},{m},{t},{dr},{ds},{dc},{dk})") )
                                        xur += xu[-1] * np.log(r_)
                                        xus += xu[-1] * np.log(s_)
                                        xuc += xu[-1] * np.log(c_)
                                        xuk += xu[-1] * np.log(k_)
                                        yu[m,t].append(val)
                                        buf_util[w,m,t] += xu[-1] * yu[m,t][-1]
                                    
                    model.addConstr(gp.quicksum(xu) == 1, name=f"C_xu_({w},{m},{t})")
                    model.addConstr(xur == factors[m,0], name=f"C_xur_({w},{m},{t})")
                    model.addConstr(xus == factors[m,1], name=f"C_xus_({w},{m},{t})")
                    model.addConstr(xuc == factors[m,4], name=f"C_xuc_({w},{m},{t})")
                    model.addConstr(xuk == factors[m,5], name=f"C_xuk_({w},{m},{t})")

            # Outputs

            t = 2
            if arch.mem_stores_datatype[m][t] or optimize_bypass_m:
                buf_util_log[w,m,t] = factors[m,2] + factors[m,3] + factors[m,5] + factors[m,6]
                
                if arch.mem_stores_multiple_datatypes[m] or optimize_bypass_m:
                    xu, xup, xuq, xuk, xun = [], 0, 0, 0, 0
                    yu[m,t] = []
                    for dp, p_ in enumerate(workload.divisors[2]):
                        for dq, q_ in enumerate(workload.divisors[3]):
                            for dk, k_ in enumerate(workload.divisors[5]):
                                for dn, n_ in enumerate(workload.divisors[6]):
                                    val = p_*q_*k_*n_
                                    if val <= arch.mem_entries[m]:
                                        xu.append( model.addVar(vtype=gp.GRB.BINARY, name=f"XU({w},{m},{t},{dp},{dq},{dk},{dn})") )
                                        xup += xu[-1] * np.log(p_)
                                        xuq += xu[-1] * np.log(q_)
                                        xuk += xu[-1] * np.log(k_)
                                        xun += xu[-1] * np.log(n_)
                                        yu[m,t].append(val)
                                        buf_util[w,m,t] += xu[-1] * yu[m,t][-1]
                                    
                                    #xu.append( model.addVar(vtype=gp.GRB.BINARY, name=f"XU({w},{m},{t},{dp},{dq},{dk},{dn})") )
                                    #xup += xu[-1] * np.log(p_)
                                    #xuq += xu[-1] * np.log(q_)
                                    #xuk += xu[-1] * np.log(k_)
                                    #xun += xu[-1] * np.log(n_)
                                    #yu[m,t].append(p_*q_*k_*n_)
                                    #buf_util[w,m,t] += xu[-1] * yu[m,t][-1]

                                    #if yu[m,t][-1] <= max_entries:
                                    #    buf_util[w,m,t] += xu[-1] * yu[m,t][-1]
                                    #else: 
                                    #    buf_util[w,m,t] += xu[-1] * (-1)
                    model.addConstr(gp.quicksum(xu) == 1, name=f"C_xu_({w},{m},{t})")
                    model.addConstr(xup == factors[m,2], name=f"C_xup_({w},{m},{t})")
                    model.addConstr(xuq == factors[m,3], name=f"C_xuq_({w},{m},{t})")
                    model.addConstr(xuk == factors[m,5], name=f"C_xuk_({w},{m},{t})")
                    model.addConstr(xun == factors[m,6], name=f"C_xun_({w},{m},{t})")

    # Constraint: respect buffer capacities -> total_buf_util <= M
    #             consider static partitioning if requested

    total_buf_util = {}
    buf_util_var = {}

    if static_partitioning:
        xs = gp.tupledict() # fraction of buffer allocated for each datatype
        for m in range(num_mems - 1):
            for t in range(num_datatypes):
                xs[m,t] = model.addVar(ub=arch.mem_entries[m], name=f'XS_({m},{t})')
            model.addConstr(xs.sum(m,'*') == arch.mem_entries[m], name=f'C_partition_({m})')

    for w in range(num_workloads):
        for m in range(num_mems - 1):
            if arch.mem_stores_multiple_datatypes[m] or (optimize_bypass and not arch.mem_bypass_defined[m]):
                if not static_partitioning:
                    total_buf_util[w,m] = 0
                    for t in range(num_datatypes):
                        buf_util_var[w,m,t] = model.addVar(ub=arch.mem_entries[m], name=f"V_buffer_util_({w},{m},{t})")
                        model.addConstr(buf_util_var[w,m,t] == buf_util[w,m,t])

                        if arch.mem_bypass_defined[m] or not optimize_bypass:
                            if arch.mem_stores_datatype[m][t]:
                                total_buf_util[w,m] += buf_util_var[w,m,t]
                        else:
                            total_buf_util[w,m] += xd[w,m,t] * buf_util_var[w,m,t]

                    model.addConstr(total_buf_util[w,m]*util_scale_factors[m] <= arch.mem_entries[m]*util_scale_factors[m], f"C_buffer_capacity_({w},{m})")
                else:
                    for t in range(num_datatypes):
                        buf_util_var[w,m,t] = model.addVar(ub=arch.mem_entries[m], name=f"V_buffer_util_({w},{m},{t})")
                        model.addConstr(buf_util_var[w,m,t] == buf_util[w,m,t])
                        
                        if arch.mem_bypass_defined[m] or not optimize_bypass:
                            if arch.mem_stores_datatype[m][t]:
                                model.addConstr(buf_util_var[w,m,t]*util_scale_factors[m] <= xs[m,t] * util_scale_factors[m], f"C_buffer_capacity_part_({w},{m},{t})")
                        else:
                            model.addConstr(xd[w,m,t] * buf_util_var[w,m,t]*util_scale_factors[m] <= xs[m,t] * util_scale_factors[m], f"C_buffer_capacity_part_({w},{m},{t})")
                        
            elif len(arch.mem_stored_datatypes[m]) == 1:
                t = arch.mem_stored_datatypes[m][0]
                model.addConstr(buf_util_log[w,m,t] <= np.log(arch.mem_entries[m]), f"C_buffer_capacity_({w},{m})_log")
                

    # Auxiliary Variable: xr_p == 1 if there is a relevant inner loop p' <= p

    if fix_permutations:
        xb_ne1 = gp.tupledict()
        for w, workload in enumerate(workloads):
            for m in range(num_mems):
                for j in range(workload.num_dims):
                    xb_ne1[w,m,j] = model.addVar(vtype=gp.GRB.BINARY, name=f'XB_NE1_({w},{m},{j})')
                    model.addConstr(xb_ne1[w,m,j] == 1 - xb[w,m,1,j,0])

    xr = gp.tupledict()
    for w, workload in enumerate(workloads):
        for t in range(num_datatypes):
            for m in range(num_mems-1):
                for m_ in range(m+1,num_mems):
                    for p, _ in enumerate(workload.bounds):
                        xr[w,t,m,m_,p] = model.addVar(vtype=gp.GRB.BINARY, name=f"XR_({w},{t},{m},{m_},{p})")

                        perm_level_sum = 0
                        for j, divs in enumerate(workload.divisors):
                            if not fix_permutations:
                                perm_level_sum += xp[w,m_,p,j] * workload.O[j][t]
                            elif workload.O[j][t] == 1:
                                #perm_level_sum += gxp[m_,p,j] * (1 - xb[w,m_,1,j,0])
                                gxp_xb_ne1 = model.addVar(vtype=gp.GRB.BINARY)
                                model.addGenConstrAnd(gxp_xb_ne1, [gxp[m_,p,j], xb_ne1[w,m_,j]])
                                perm_level_sum += gxp_xb_ne1

                        if m_ == m+1 and p == 0:
                            model.addConstr(xr[w,t,m,m_,p] == perm_level_sum, name=f"C_xr_pls_({w},{t},{m},{m_},{p})")
                        elif m_ > m+1 and p == 0:
                            model.addConstr(xr[w,t,m,m_,p] >= xr[w,t,m,m_-1,len(workload.bounds)-1], name=f"C_xp_gt_({w},{t},{m},{m_},{p})")
                            model.addConstr(xr[w,t,m,m_,p] >= perm_level_sum, name=f"C_xr_pls_({w},{t},{m},{m_},{p})")
                        elif p > 0: 
                            model.addConstr(xr[w,t,m,m_,p] >= xr[w,t,m,m_,p-1])
                            model.addConstr(xr[w,t,m,m_,p] >= perm_level_sum, name=f"C_xr_pls_({w},{t},{m},{m_},{p})")

    # Auxiliary Variable: xj_j == 1 if j loop has inner relevant loop

    xj = {}
    for w, workload in enumerate(workloads):
        for t in range(num_datatypes):
            for m in range(num_mems-1):
                for m_ in range(m+1,num_mems):
                    for j, _ in enumerate(workload.bounds):
                        xj[w,t,m,m_,j] = model.addVar(vtype=gp.GRB.BINARY, name=f"YB_({w},{t},{m},{m_},{j})")
                        prod = 0
                        for p, _ in enumerate(workload.bounds):
                            #prod += xp[l,m_,p,j] * yp[l,v,m,m_,p]
                            xp_yp = model.addVar(vtype=gp.GRB.BINARY, name=f"AUX_XP_YP({w},{t},{m},{m_},{j},{p})")
                            if not fix_permutations:
                                model.addGenConstrAnd(xp_yp, [xp[w,m_,p,j], xr[w,t,m,m_,p]], name=f"AND_XP_YP_({w},{t},{m},{m_},{j},{p})")
                            else:
                                model.addGenConstrAnd(xp_yp, [gxp[m_,p,j], xr[w,t,m,m_,p]], name=f"AND_GXP_YP_({w},{t},{m},{m_},{j},{p})")
                            prod += xp_yp
                        model.addConstr(xj[w,t,m,m_,j] == prod, name=f"C_yb_({w},{t},{m},{m_},{j})")
    
    # Expression: memory writes
    
    mem_writes_inst = {} # per-instance
    mem_writes = {} # total
    for w, workload in enumerate(workloads):
        for m in range(num_mems-1):
            for t in range(num_datatypes):
                mem_writes_inst[w,m,t] = mem_writes[w,m,t] = 0
                if not arch.mem_stores_datatype[m][t] and not (optimize_bypass and not arch.mem_bypass_defined[m]):
                    continue
                
                # Buf util ultiplied by relevant or outer than relevant loops
                mem_writes_inst[w,m,t] = gp.LinExpr(buf_util_log[w,m,t])
                for m_ in range(m+1,num_mems):
                    for j, divs in enumerate(workload.divisors):
                        for i, div in enumerate(divs):
                            xb_xj = model.addVar(vtype=gp.GRB.BINARY, name=f"V_and_xb_xj_({w},{t},{m},{m_},{j},{i})")
                            model.addGenConstrAnd(xb_xj, [xb[w,m_,1,j,i], xj[w,t,m,m_,j]], name=f"C_and_xb_xj_({w},{t},{m},{m_},{j},{i})")
                            mem_writes_inst[w,m,t] += xb_xj * np.log(div)
                # Multiply by num of instances (prod of all spatial loops above mem)
                mem_writes[w,m,t] = gp.LinExpr(mem_writes_inst[w,m,t])
                for m_ in range(m+1,num_mems):
                    for j, divs in enumerate(workload.divisors):
                        for i, div in enumerate(divs):
                            mem_writes[w,m,t] += xb[w,m_,0,j,i] * np.log(div)
            
    # Expression: memory reads

    mem_reads_inst = {}
    mem_reads = {}

    if not optimize_bypass:
        for w, workload in enumerate(workloads):
            macs = np.log(workload.macs)
            for t in range(num_datatypes):
                m_ = -1 # Inner buffer containing same datatype
                for m in range(num_mems):
                    if arch.mem_stores_datatype[m][t]:
                        if m_ == -1:
                            mem_reads_inst[w,m,t] = mem_reads[w,m,t] = macs
                            for m__ in range(m+1,num_mems):
                                for j, divs in enumerate(workload.divisors):
                                    for i, div in enumerate(divs):
                                        mem_reads_inst[w,m,t] -= xb[w,m__,0,j,i] * np.log(div)
                            for m__ in range(0,m+1):
                                for j, divs in enumerate(workload.divisors):
                                    for i, div in enumerate(divs):
                                        isl = xb[w,m__,0,j,i] * (1 - workload.O[j][t]) * np.log(div)    
                                        mem_reads_inst[w,m,t] -= isl
                                        mem_reads[w,m,t] -= isl
                        else:
                            mem_reads_inst[w,m,t] = gp.LinExpr(mem_writes_inst[w,m_,t])
                            # Multiply writes by spatial relevant loops between write buffer (excl) and read buffer (incl)
                            for m__ in range(m_+1, m+1):
                                for j, divs in enumerate(workload.divisors):
                                    for i, div in enumerate(divs):
                                        mem_reads_inst[w,m,t] += xb[w,m__,0,j,i] * workload.O[j][t] * np.log(div)
                                
                            # Multibly by all spatial loops above read buffer (excl)
                            mem_reads[w,m,t] = gp.LinExpr(mem_reads_inst[w,m,t])
                            for m__ in range(m+1,num_mems):
                                for j, divs in enumerate(workload.divisors):
                                    for i, div in enumerate(divs):
                                        mem_reads[w,m,t] += xb[w,m__,0,j,i] * np.log(div)
                        m_ = m
                    else:
                        mem_reads[w,m,t] = mem_reads_inst[w,m,t] = 0
    else:
        for w, workload in enumerate(workloads):
            macs = np.log(workload.macs)
            for t in range(num_datatypes):
                for m in range(num_mems):

                    mem_reads_inst[w,m,t] = gp.LinExpr(0)
                    mem_reads[w,m,t] = gp.LinExpr(0)
                
                    for m_ in range(-1, m):
                        orl = [xd[w,m__,t] for m__ in range(m_)] + [xd[w,m__,t] for m__ in range(m_+1, m)]
                        orv = model.addVar(vtype=gp.GRB.BINARY, name=f"V_and_xb_xj_({w},{t},{m},{m_})")
                        model.addGenConstrOr(orv, orl)
                        if m_ == -1:
                            orf = model.addVar(vtype=gp.GRB.BINARY)
                            norv = model.addVar(vtype=gp.GRB.BINARY)
                            model.addConstr(norv == 1 - orv)
                            model.addGenConstrAnd(orf, [norv, xd[w,m,t]])

                            mem_reads_inst[w,m,t] += orf * macs
                            mem_reads[w,m,t] += orf * macs

                            for m__ in range(m+1,num_mems):
                                for j, divs in enumerate(workload.divisors):
                                    for i, div in enumerate(divs):
                                        mem_reads_inst[w,m,t] -= xb[w,m__,0,j,i] * np.log(div) * orf
                            for m__ in range(0,m+1):
                                for j, divs in enumerate(workload.divisors):
                                    for i, div in enumerate(divs):
                                        isl = xb[w,m__,0,j,i] * (1 - workload.O[j][t]) * np.log(div) * orf  
                                        mem_reads_inst[w,m,t] -= isl
                                        mem_reads[w,m,t] -= isl
                        else:
                            orf = model.addVar(vtype=gp.GRB.BINARY)
                            norv = model.addVar(vtype=gp.GRB.BINARY)
                            model.addConstr(norv == 1 - orv)
                            model.addGenConstrAnd(orf, [norv, xd[w,m_,t], xd[w,m,t]])

                            mem_reads_inst[w,m,t] += orf * mem_writes_inst[w,m_,t]
                            mem_reads[w,m,t] += orf * mem_writes_inst[w,m_,t]

                            # Multiply writes by spatial relevant loops between write buffer (excl) and read buffer (incl)
                            for m__ in range(m_+1, m+1):
                                for j, divs in enumerate(workload.divisors):
                                    for i, div in enumerate(divs):
                                        mem_reads_inst[w,m,t] += xb[w,m__,0,j,i] * workload.O[j][t] * np.log(div) * orf 
                                
                            # Multibly by all spatial loops above read buffer (excl)
                            for m__ in range(m+1,num_mems):
                                for j, divs in enumerate(workload.divisors):
                                    for i, div in enumerate(divs):
                                        mem_reads[w,m,t] += xb[w,m__,0,j,i] * np.log(div) * orf

    # Objective: memory access energy

    est_buf_energy = {}
    total_energy = 0
    pwl_opts = "FuncPieces=-2 FuncPieceError=0.002"

    for w, workload in enumerate(workloads):
        for t in range(num_datatypes):
            for m in range(num_mems):
                if not arch.mem_stores_datatype[m][t] and not (optimize_bypass and not arch.mem_bypass_defined[m]):
                    est_buf_energy[w,m,t] = 0
                    continue

                lb = 0.98*arch.mem_access_cost[m]*macs_scale_factors[w]
                ub = 1.02*workload.macs*arch.mem_access_cost[m]*macs_scale_factors[w]

                write_energy = 0
                if m < num_mems - 1:
                    
                    log_write_energy = model.addVar(lb=np.log(lb), ub=np.log(ub))
                    model.addConstr(log_write_energy == mem_writes[w,m,t] + np.log(arch.mem_access_cost[m]) + macs_scale_factor_logs[w])
                    
                    write_energy = model.addVar(ub=ub)
                    model.addGenConstrExp(log_write_energy, write_energy, options=pwl_opts, name=f"C_exp_write_energy_{w}_{m}_{t}")
                
                log_read_energy = model.addVar(lb=np.log(lb), ub=np.log(ub))
                model.addConstr(log_read_energy == mem_reads[w,m,t] + np.log(arch.mem_access_cost[m]) + macs_scale_factor_logs[w])
                
                read_energy = model.addVar(ub=ub)
                model.addGenConstrExp(log_read_energy, read_energy, options=pwl_opts, name=f"C_exp_read_energy_{w}_{m}_{t}")

                if arch.mem_bypass_defined[m] or not optimize_bypass:
                    if arch.mem_stores_datatype[m][t]:
                        est_buf_energy[w,m,t] = (write_energy + read_energy)
                    else:
                        est_buf_energy[w,m,t] = 0
                else:
                    est_buf_energy[w,m,t] = (write_energy + read_energy) * xd[w,m,t]
                
                total_energy += workload.weight * (est_buf_energy[w,m,t] / macs_scale_factors[w]) * max_macs_scale_factor
        
        total_energy += workload.weight * arch.mac_energy * workload.macs * max_macs_scale_factor

    
    # Expression / objective: compute cycles

    total_compute = 0
    compute_cycles = {}
    pwl_opts = "FuncPieces=-2 FuncPieceError=0.002"

    for w, workload in enumerate(workloads):
        log_compute_cycles_expr = 0
        for m in range(num_mems):
            for j, divs in enumerate(workload.divisors):
                for i, div in enumerate(divs):
                    log_compute_cycles_expr += xb[w,m,1,j,i] * np.log(div)

        ub = 1.02*workload.macs*macs_scale_factors[w]

        log_compute_cycles = model.addVar(lb = macs_scale_factor_logs[w], ub = np.log(ub))
        model.addConstr(log_compute_cycles == log_compute_cycles_expr + macs_scale_factor_logs[w])
        
        compute_cycles[w] = model.addVar(ub = ub)
        model.addGenConstrExp(log_compute_cycles, compute_cycles[w], options=pwl_opts, name=f"C_exp_compute_cycles_{w}")
        
        total_compute += workload.weight * (compute_cycles[w] / macs_scale_factors[w]) * max_macs_scale_factor

    
    # Expression: total memory reads per memory (all datatypes)
    
    tot_mem_reads_inst = {}
    pwl_opts = "FuncPieces=-2 FuncPieceError=0.002"

    for w, workload in enumerate(workloads):
        for m in range(num_mems):
            tot_mem_reads_inst[w,m] = 0
            for t in range(num_datatypes):
                ub = 1.02*workload.macs*macs_scale_factors[w]

                log_reads = model.addVar(lb = macs_scale_factor_logs[w], ub=np.log(ub))
                model.addConstr(log_reads == mem_reads_inst[w,m,t] + macs_scale_factor_logs[w], name=f"C_log_reads_{w}_{m}_{t}") # inst or not 
                
                reads = model.addVar(ub=ub)
                model.addGenConstrExp(log_reads, reads, options=pwl_opts, name=f"C_exp_reads_{w}_{m}_{t}")

                if arch.mem_bypass_defined[m] or not optimize_bypass:
                    if arch.mem_stores_datatype[m][t]:
                        tot_mem_reads_inst[w,m] += reads
                else:
                    tot_mem_reads_inst[w,m] += reads * xd[w,m,t]

    # Objective: simulation latency

    total_latency = 0
    memory_cycles = {}
    latency = {}
    for w, workload in enumerate(workloads):
        bottleneck_candidates = [compute_cycles[w]]
        for m in range(num_mems):
            memory_cycles[w,m] = model.addVar(ub=1.02*workload.macs*macs_scale_factors[w], name=f"V_mem_cycles_({w},{m})") # *
            model.addConstr(memory_cycles[w,m] == tot_mem_reads_inst[w,m] / arch.read_bandwidth[m])
            bottleneck_candidates.append(memory_cycles[w,m])
        
        latency[w] = model.addVar(ub=1.02*workload.macs*macs_scale_factors[w]) # *
        model.addGenConstrMax(latency[w], bottleneck_candidates, name=f"max_latency_{w}")
        total_latency += workload.weight * (latency[w] / macs_scale_factors[w]) * max_macs_scale_factor

        # *: mmmh not true if there is some bandwidth < 1


    # Optimize

    model.ModelSense = gp.GRB.MINIMIZE

    if objective_type == 'blended':
        objective = latency_weight * total_latency + energy_weight * total_energy
        model.setObjective(objective, gp.GRB.MINIMIZE)
        print(f"Setting the objective: blended ({latency_weight}*LATENCY + {energy_weight}*ENERGY)")
    elif objective_type == 'latency-energy':
        model.setObjectiveN(total_latency, 0, 1, name="latency")
        model.setObjectiveN(total_energy, 1, 0, name="energy")
        print(f"Setting the objective: hierarchical (LATENCY->ENERGY)")
    elif objective_type == 'energy-latency':
        model.setObjectiveN(total_energy, 0, 1, name="energy")
        model.setObjectiveN(total_latency, 1, 0, name="latency")
        print(f"Setting the objective: hierarchical (ENERGY->LATENCY)")
    elif objective_type == 'quadratic':
        model.setParam('NonConvex', 2)
        total_latency_var = model.addVar(name="total_latency")
        model.addConstr(total_latency_var == total_latency)
        total_energy_var = model.addVar(name="total_energy")
        model.addConstr(total_energy_var == total_energy)
        model.setObjective(total_latency_var * total_energy_var, gp.GRB.MINIMIZE)
        print(f"Setting the objective: quadratic (ENERGY*LATENCY)")
    else:
        sys.exit('🔴 Error: invalid obj type')

    #model.computeIIS()
    #model.write("model.ilp")

    model.write("model.lp")

    model_build_time = time.time() - model_build_time
    print(f"\n⚪ MIP model built in {model_build_time:.3f} s. Optimizing...\n")

    if num_workloads == 1:
        if not optimize_bypass:
            time_threshold = 60
            #time_limit = 200
            time_limit = time_limit_single
            mip_gap = 0.025
            def termination(model, where):
                if where == gp.GRB.Callback.MIP:
                    runtime = model.cbGet(gp.GRB.Callback.RUNTIME)
                    objbst = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
                    objbnd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
                    gap = abs((objbst - objbnd) / objbst)

                    if runtime > time_threshold and gap <= mip_gap:
                        model.terminate()

            model.setParam('TimeLimit', time_limit)
            model.optimize(termination)
        else:
            #model.setParam('MIPGap', 0.005)
            model.optimize()
    else:
        model.setParam('MIPGap', 0.005)
        model.optimize()

    #model.optimize()
    #model.write("start.mst")

    if model.SolCount == 0:
        sys.exit("\n😔 No solutions found.\n")

    model.printQuality()
    print(f"\n🟢 Optimization completed in {model.Runtime} s")
    print(f'   Energy Obj.: {total_energy.getValue()} - Latency Obj.: {total_latency.getValue()}')

    # Translate solution to mapping configs

    layer_mappings = []
    for w, workload in enumerate(workloads):

        bounds, permutations, unrollings, bypass = [], [], [], []

        for m in range(num_mems):
            # Permutations
            rank_to_dim = []
            if not fix_permutations and not old_fp:
                for p in workload.dim_idxs:
                    for j in workload.dim_idxs:
                        if xp[w,m,p,j].X > 0.9:
                            rank_to_dim.append(j)
                rank_to_dim.extend(list(set(workload.dim_idxs).difference(set(rank_to_dim))))
            else:
                for p in workload.dim_idxs:
                    for j in workload.dim_idxs:
                        if gxp[m,p,j].X > 0.9:
                            rank_to_dim.append(j)
            permutations.append(rank_to_dim)

            # Bounds and Unrollings
            m_bounds, m_unrollings = [], []
            for j, divs in enumerate(workload.divisors):
                for i, div in enumerate(divs):
                    if xb[w,m,1,j,i].X > 0.9:
                        m_bounds.append(div)
                    if xb[w,m,0,j,i].X > 0.9:
                        m_unrollings.append(div)
            bounds.append(m_bounds)
            unrollings.append(m_unrollings)

            m_bypass = [xd[w,m,t].X > 0.9 for t in range(num_datatypes)]
            bypass.append(m_bypass)

        layer_mappings.append(Mapping(workload, arch, bounds, permutations, unrollings, bypass))

        # Debug Mapping Print 

        print(f'\n   Workload {w}\n    Mapping: ', end='')
        for m in range(num_mems):
            print(f'[L{m}] ', end='') 
            for j in permutations[m]:
                if bounds[m][j] != 1:
                    dim_name = workload.dim_idx_name_dict[j]
                    print(f'{dim_name}{bounds[m][j]} ', end='') 

        dt_name = {0: 'I', 1: 'W', 2: 'O'}

        print('\n    Est. Writes: ', end='')
        for m in range(num_mems-1):
            print(f'[L{m}] ', end='') 
            for t in range(num_datatypes):
                writes = np.exp(getattr(mem_writes_inst[w,m,t], 'getValue', lambda : float('-Inf'))())
                print(f'{dt_name[t]}:{round(writes)} ', end='') 
        
        print('\n    Est. Reads: ', end='')
        for m in range(num_mems):
            print(f'[L{m}] ', end='') 
            for t in range(num_datatypes):
                reads = np.exp(getattr(mem_reads_inst[w,m,t], 'getValue', lambda : float('-Inf'))())
                print(f'{dt_name[t]}:{round(reads)} ', end='') 

        print('\n    Buf. util: ', end='')
        for m in range(num_mems):
            print(f'[L{m}] ', end='') 
            for t in range(num_datatypes):
                util = getattr(buf_util[w,m,t], 'getValue', lambda : 0)()
                print(f'{dt_name[t]}:{round(util)} ', end='') 

        print('\n    Buf. util log: ', end='')
        for m in range(num_mems):
            print(f'[L{m}] ', end='') 
            for t in range(num_datatypes):
                util_log = np.exp(getattr(buf_util_log[w,m,t], 'getValue', lambda : float('-Inf'))())
                print(f'{dt_name[t]}:{round(util_log)} ', end='') 

        print('\n', end='')
            

    return (layer_mappings, model.Runtime)

