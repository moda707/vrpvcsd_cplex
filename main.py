import copy
import json
import math
import random
import sys
import numpy as np
from docplex.mp.model import Model
from scipy.spatial.distance import pdist, squareform

import instance_generator
import vrp
from vrp import VRPSimulator


# index encoder and decoders to convert multiple dimensions to one dimension vector
def ind_encode2(i, j, n1, n2):
    return i + j * n1


def ind_decode2(i, n1, n2):
    return i % n1, i // n1


def ind_encode3(i, j, k, n1, n2, n3):
    return i + j * n1 + k * n1 * n2


def ind_decode3(i, n1, n2, n3):
    return i % n1, (i // n1) % n2, i // (n1 * n2)


def ind_encode4(i, j, k, w, n1, n2, n3, n4):
    return i + j * n1 + k * n1 * n2 + w * n1 * n2 * n3


def ind_decode4(i, n1, n2, n3, n4):
    return i % n1, (i // n1) % n2, (i // (n1 * n2)) % n3, i // (n1 * n2 * n3)


def compute_w(customers, L, Q, depot=(35, 35)):
    """
    Computes an upper bound for the number of trips
    :param customers: set of customers
    :param L: Duration limit
    :param Q: Capacity of the vehicle
    :param depot: location of the depot
    :return:
    """
    f = [[c[4], math.sqrt((c[1] - depot[0]) ** 2 + (c[2] - depot[1]) ** 2)] for c in customers]

    f = sorted(f, key=lambda x: x[1])
    rl = 0
    w = 0
    i = 0
    while rl <= L:
        if rl + 2 * f[i][1] > L:
            return w
        else:
            rl += 2 * f[i][1]
            w += 1
            if f[i][0] > Q:
                f[i][0] -= Q
            else:
                i += 1


def cplex_model(customers, scenario, m, Q, L, depot):
    """
    Detailed mathematical model is available in the original paper

    :param customers: set of customers
    :param scenario: a set of demand scenarios
    :param m: number of vehicles
    :param Q: capacity
    :param L: duration limit
    :param depot: location of the depot
    :return:
    """
    M = 100000000
    N = len(customers) + 1
    K = m

    distance_coef = 0.0001

    nodes = np.zeros([len(customers) + 1, 2])
    nodes[0] = depot
    nodes[1:] = customers[:, 1:3]
    d = np.zeros(len(customers) + 1)
    d[1:] = scenario
    # print("Total demand:", sum(d))

    t = squareform(pdist(np.array(nodes[:, :2])))

    # compute w
    W = compute_w(customers, L, Q, depot)

    m = Model(name='mathmodel')
    m.parameters.timelimit = 3600
    m.set_time_limit(3600)

    m.parameters.emphasis.mip = 1

    # m.variables.
    # variables
    x = m.continuous_var_list(N * K * W, lb=0, name="x_ikw")
    y = m.binary_var_list(N * N * K * W, name="y_ijkw")
    l_ikw = m.binary_var_list(N * K * W, name="lambda_ikw")
    q_ikw = m.continuous_var_list(N * K * W, lb=0, ub=Q, name="q_ikw")
    t_ikw = m.continuous_var_list(N * K * W, lb=0, ub=L, name="t_ikw")
    ts_kw = m.continuous_var_list(K * W, lb=0, ub=L, name="ts_kw")

    obj_func = sum(x[ind_encode3(i, k, w, N, K, W)] for i in range(1, N) for k in range(K) for w in range(W))
    obj_func -= distance_coef * sum(ts_kw[ind_encode2(k, W - 1, K, W)] +
                                    sum(t[i, j] * y[ind_encode4(i, j, k, W - 1, N, N, K, W)]
                                        for i in range(N)
                                        for j in range(N))
                                    for k in range(K)) / K
    m.set_objective("max", obj_func)

    # constraints
    for i in range(N):
        for k in range(K):
            for w in range(W):
                # const 1*
                c1 = sum(y[ind_encode4(i, j, k, w, N, N, K, W)] for j in range(N))
                m.add_constraint(c1 <= 1, ctname=f"ct1_{i}_{k}_{w}")
                # const 2*
                c2 = sum(y[ind_encode4(j, i, k, w, N, N, K, W)] for j in range(N))
                m.add_constraint(c2 <= 1, ctname=f"ct2_{i}_{k}_{w}")

                # const 3*
                m.add_constraint(y[ind_encode4(i, i, k, w, N, N, K, W)] == 0, ctname=f"ct3_{i}_{k}_{w}")

                # const 5*
                m.add_constraint(c2 == c1, ctname=f"ct5_{i}_{k}_{w}")
                m.add_constraint(c2 == l_ikw[ind_encode3(i, k, w, N, K, W)])

                if i > 0:
                    # const 8*
                    m.add_constraint(x[ind_encode3(i, k, w, N, K, W)] <= M * l_ikw[ind_encode3(i, k, w, N, K, W)],
                                     ctname=f"ctp_{i}_{k}_{w}")
                    # const 11
                    # m.add_constraint(x[ind_encode3(i, k, w, N, K, W)] <= d_iw[ind_encode2(i, w, N, W)],
                    #                  ctname=f"ct11_{i}_{k}_{w}")

                    # const 11*
                    m.add_constraint(x[ind_encode3(i, k, w, N, K, W)] <= q_ikw[ind_encode3(i, k, w, N, K, W)],
                                     ctname=f"ct12_{i}_{k}_{w}")

                    # const 13
                    # m.add_constraint(x[ind_encode3(i, k, w, N, K, W)] >= d_iw[ind_encode2(i, w, N, W)] - M * (
                    #         1 - tmp_1[ind_encode3(i, k, w, N, K, W)]), ctname=f"ct13_{i}_{k}_{w}")

                    # const 14
                    # m.add_constraint(
                    #     x[ind_encode3(i, k, w, N, K, W)] >= q_ikw[ind_encode3(i, k, w, N, K, W)] - M * tmp_1[
                    #         ind_encode3(i, k, w, N, K, W)], ctname=f"ct14_{i}_{k}_{w}")

                    # const 13*
                    m.add_constraint(
                        q_ikw[ind_encode3(i, k, w, N, K, W)] <= Q + M * (1 - y[ind_encode4(0, i, k, w, N, N, K, W)]),
                        ctname=f"ct18_{i}_{k}_{w}")

                    # const 14*
                    m.add_constraint(
                        q_ikw[ind_encode3(i, k, w, N, K, W)] >= Q - M * (1 - y[ind_encode4(0, i, k, w, N, N, K, W)]),
                        ctname=f"ct19_{i}_{k}_{w}")

                    for j in range(1, N):
                        # const 15*
                        m.add_constraint(q_ikw[ind_encode3(i, k, w, N, K, W)] <= (
                                q_ikw[ind_encode3(j, k, w, N, K, W)] - x[ind_encode3(j, k, w, N, K, W)]) + M * (
                                                 1 - y[ind_encode4(j, i, k, w, N, N, K, W)]),
                                         ctname=f"ct20_{i}_{k}_{w}")

                        # const 16*
                        m.add_constraint(q_ikw[ind_encode3(i, k, w, N, K, W)] >= (
                                q_ikw[ind_encode3(j, k, w, N, K, W)] - x[ind_encode3(j, k, w, N, K, W)]) - M * (
                                                 1 - y[ind_encode4(j, i, k, w, N, N, K, W)]),
                                         ctname=f"ct21_{i}_{k}_{w}")

                    # const 17*
                    m.add_constraint(
                        q_ikw[ind_encode3(i, k, w, N, K, W)] <= M * l_ikw[ind_encode3(i, k, w, N, K, W)],
                        ctname=f"ct22_{i}_{k}_{w}")

                    # const 20*
                    m.add_constraint(
                        t_ikw[ind_encode3(i, k, w, N, K, W)] >= ts_kw[ind_encode2(k, w, K, W)] + t[0, i] - M * (
                                1 - y[ind_encode4(0, i, k, w, N, N, K, W)]), ctname=f"ct25_{i}_{k}_{w}")

                    # const 21*
                    for j in range(1, N):
                        m.add_constraint(t_ikw[ind_encode3(i, k, w, N, K, W)] >= (
                                t_ikw[ind_encode3(j, k, w, N, K, W)] + t[j, i]) - M * (
                                                 1 - y[ind_encode4(j, i, k, w, N, N, K, W)]),
                                         ctname=f"ct26_{i}_{k}_{w}")
                        # to make the constraint above as equality
                        m.add_constraint(t_ikw[ind_encode3(i, k, w, N, K, W)] <= (
                                t_ikw[ind_encode3(j, k, w, N, K, W)] + t[j, i]) + M * (
                                                 1 - y[ind_encode4(j, i, k, w, N, N, K, W)]),
                                         ctname=f"ct26_{i}_{k}_{w}")

                    # const 22*
                    m.add_constraint(t_ikw[ind_encode3(i, k, w, N, K, W)] <= M * l_ikw[ind_encode3(i, k, w, N, K, W)],
                                     ctname=f"ct27_{i}_{k}_{w}")

            # const 6
            # if i > 0:
            #     m.add_constraint(sum(l_ikw[ind_encode3(i, k, w1, N, K, W)] for w1 in range(W)) <=
            #                      l_ik[ind_encode2(i, k, N, K)] * W, ctname=f"ct6_{i}_{k}")

        # const 10*
        m.add_constraint(sum(x[ind_encode3(i, k1, w1, N, K, W)] for k1 in range(K) for w1 in range(W)) <= d[i])

    for k in range(K):
        # const 18*
        m.add_constraint(ts_kw[ind_encode2(k, 0, K, W)] == 0, ctname=f"ct23_{k}")

        # const 23*
        m.add_constraint(ts_kw[ind_encode2(k, W - 1, K, W)] + sum(
            t[i, j] * y[ind_encode4(i, j, k, W - 1, N, N, K, W)] for i in range(N) for j in range(N)) <= L,
                         ctname=f"ct28_{k}_{W - 1}")

        for w in range(W):
            # const 4*
            m.add_constraint(sum(y[ind_encode4(0, i, k, w, N, N, K, W)] for i in range(N)) <= sum(
                l_ikw[ind_encode3(i, k, w, N, K, W)] for i in range(N)), ctname=f"ct3_{k}_{w}")

            # const 12*
            m.add_constraint(sum(x[ind_encode3(i, k, w, N, K, W)] for i in range(N)) <= Q, ctname=f"ct15_{k}_{w}")

            # const 9*
            m.add_constraint(x[ind_encode3(0, k, w, N, K, W)] == 0, ctname=f"ct10_{k}_{w}")
            if w > 0:
                # const 6*
                m.add_constraint(sum(y[ind_encode4(i1, j1, k, w, N, N, K, W)]
                                     for i1 in range(N) for j1 in range(N)) <=
                                 sum(y[ind_encode4(i1, j1, k, w - 1, N, N, K, W)]
                                     for i1 in range(N) for j1 in range(N)) * M)

                # const 7*
                m.add_constraint(sum(l_ikw[ind_encode3(i1, k, w, N, K, W)] for i1 in range(N)) <= M * sum(
                    l_ikw[ind_encode3(i1, k, w - 1, N, K, W)] for i1 in range(1, N)), ctname=f"ct8_{k}_{w}")

                # const 19*
                expr = ts_kw[ind_encode2(k, w - 1, K, W)] + sum(
                    t[i, j] * y[ind_encode4(i, j, k, w - 1, N, N, K, W)] for i in range(N) for j in range(N))
                m.add_constraint(ts_kw[ind_encode2(k, w, K, W)] == expr, ctname=f"ct24_{k}_{w}")

    # m.print_information()
    m.solve()

    routes = []
    served_record = []
    arrival_record = []
    coord_record = []
    for k in range(K):
        route_k = [0]
        served_k = [0]
        route_at = [0]
        coord_k = [[35, 35]]
        for w in range(W):
            current_node = 0
            # find next node
            while True:
                ii = None
                for i in range(N):
                    if y[ind_encode4(current_node, i, k, w, N, N, K, W)].solution_value == 1:
                        ii = i
                        break
                if ii is None:
                    break
                next_node = ii
                route_k.append(next_node)
                served_k.append(x[ind_encode3(next_node, k, w, N, K, W)].solution_value)
                route_at.append(t_ikw[ind_encode3(next_node, k, w, N, K, W)].solution_value)
                coord_k.append(nodes[next_node].tolist())
                current_node = next_node
                if current_node == 0:
                    break
        routes.append(route_k)
        served_record.append(served_k)
        arrival_record.append(route_at)
        coord_record.append(coord_k)

    rvalue = distance_coef * sum(ts_kw[ind_encode2(k, W - 1, K, W)].solution_value +
                                 sum(t[i, j] * y[ind_encode4(i, j, k, W - 1, N, N, K, W)].solution_value
                                     for i in range(N)
                                     for j in range(N)) for k in range(K)) / K
    print("rvalue=", rvalue)
    real_obj_value = m.objective_value + rvalue

    print("demands:", d)
    print("served demands:", served_record)
    print("arrival times:", arrival_record)
    print("visits:", coord_record)

    print(f"Exp. Dem:{sum(d):.2f}\tW:{W}\t"
          f"Obj. Value:{real_obj_value:.2f}\t"
          f"time:{m.solve_details.time:.3f}\troute:{routes}\tGap:{m.solve_details.gap * 100:.4f}%")
    print(m.objective_value, m.solve_details.best_bound, m.solve_details.gap, m.solve_details.mip_relative_gap)


def cplex_solver(customers_set, demand_scenarios, m=2, L=None, Q=None, start=0):
    # customers_set = list(customers_set)[6:7]
    # demand_scenarios = list(demand_scenarios)[6:7]
    for ins, customersset in enumerate(customers_set):
        if ins < start:
            continue
        print("Instance", ins, ":")
        customers, n_real = customersset
        # solve for the mean demand
        print("Average demand:")
        mean_scenario = np.array(customers)[:n_real, -1]
        cplex_model(np.array(customers[:n_real]), mean_scenario, m, Q, L, [50, 50])


def demand_realization(exp_dem):
    c = [0, .5, 1., 1.5, 2.]
    p = [0.05, 0.15, 0.6, 0.15, 0.05]
    return np.random.choice(c, p=p) * exp_dem

    # if exp_dem == 5:
    #     return random.choice(list(range(1, 10)))
    # elif exp_dem == 10:
    #     return random.choice(list(range(5, 16)))
    # elif exp_dem == 15:
    #     return random.choice(list(range(10, 21)))


def fixed_route_evaluator(n, capacity, duration_limit):
    env_args = {"service_area": [100, 100],
                "xy_steps": [10, 10],
                "model_type": "VRPSCD",
                "m": 2}
    # capacity = 50.
    # duration_limit = 201.38
    env_config = instance_generator.EnvConfig(**env_args)
    vrp_env = vrp.VRPSD(env_config)
    vrpsim = VRPSimulator(vrp_env)

    fixed_routes = [[[0, 4, 12, 11, 13, 14, 0], [0, 6, 7, 2, 5, 0, 8, 9, 10, 0]],
                    [[0, 5, 7, 8, 0, 12, 11, 13, 0], [0, 4, 2, 1, 3, 0]],
                    [[0, 14, 5, 4, 9, 0, 16, 17, 15, 13, 0], [0, 11, 7, 6, 1, 2, 3, 8, 0, 12, 0]],
                    [[0, 1, 3, 2, 4, 0], [0, 10, 6, 5, 11, 0, 9, 8, 7, 12, 0]],
                    [[0, 6, 5, 3, 2, 4, 1, 0], [0, 10, 8, 7, 9, 6, 0]],
                    [[0, 15, 17, 16, 0, 11, 12, 13, 10, 0], [0, 4, 2, 1, 6, 0, 9, 0]],
                    [[0, 2, 6, 8, 7, 0, 10, 0], [0, 11, 12, 16, 17, 18, 0, 9, 13, 14, 0]],
                    [[0, 11, 8, 7, 4, 1, 5, 0, 9, 0], [0, 10, 6, 3, 0, 12, 15, 16, 13, 0]],
                    [[0, 11, 7, 8, 4, 3, 0, 9, 12, 10, 0], [0, 2, 1, 5, 6, 14, 13, 0]],
                    [[0, 4, 5, 6, 7, 12, 11, 9, 0], [0, 3, 1, 2, 4, 0, 8, 9, 10, 11, 0]]]
    # fixed_routes = [[[0, 6, 8, 9, 7, 12, 0], [0, 2, 1, 3, 4, 5, 0, 5, 0]]]

    with open(f"Instances/VRPSCD/t_{n}_10") as file:
        test_customers_set = json.load(file).values()
    instances = []
    instance_args = {"n": n,
                     "m": 2,
                     "capacity": capacity,
                     "duration_limit": duration_limit,
                     "real_duration_limit": duration_limit,
                     "density_class": 0,
                     "stoch_type": 2,
                     "depot": [50, 50]}
    instance_config = instance_generator.InstanceConfig(**instance_args)
    for customers in test_customers_set:
        test_vehicles = np.array([[j, 50, 50,
                                   capacity, 0, len(customers[0])]
                                  for j in range(2)])
        test_customers = np.array(customers[0])
        # test_customers[:, [1, 2]] /= Utils.Norms.COORD
        # test_customers[:, [4, 8]] /= instance_config.capacity
        ins_config = copy.copy(instance_config)
        ins_config.real_n = customers[1]
        ins_config.duration_limit = duration_limit
        ins_config.m = 2
        ins_config.n = len(test_customers)
        ins_config.capacity = ins_config.capacity
        instance = {"Customers": test_customers,
                    "Vehicles": test_vehicles,
                    "Config": ins_config,
                    "Name": random.randint(1000, 9999)}
        instances.append(instance)
    with open(f"Instances/VRPSCD/realization_{n}_100_n", "r") as file:
        scenarios_set = list(json.load(file).values())
    for eins, instance in enumerate(instances):
        # if eins > 4:
        #     continue
        scenarios = scenarios_set[eins]
        # plot_customers(instance["Customers"], service_area=100)
        avg_scenario = instance["Customers"][:, -1]
        nn0 = [[50, 50] if i == 0 else instance["Customers"][i - 1, 1:3] for i in fixed_routes[eins][0]]
        nn1 = [[50, 50] if i == 0 else instance["Customers"][i - 1, 1:3] for i in fixed_routes[eins][1]]
        # rr0 = vrp.VRPSD.get_seq_distance(nn0)
        # rr1 = vrp.VRPSD.get_seq_distance(nn1)
        # results = vrpsim.simulate_fixed_route(instance, avg_scenario, fixed_routes[eins])
        # print(f"{results.expected_demand}\t{results.final_reward}")
        for esc, scenario in enumerate(scenarios):
            # if esc > 4:
            #     continue
            results = vrpsim.simulate_fixed_route(instance, scenario, fixed_routes[eins])
            # print(f"Instance {eins} Scenario {esc}: Obj:{results.final_reward}")
            print(f"{results.expected_demand}\t{results.final_reward}")
        print("-----------------------")


if __name__ == "__main__":
    fixed_route_evaluator(15, 50, 201.38)
    quit()
    args = sys.argv
    if len(args) > 1:
        _, n, m, L, Q, st = args
    else:
        n = 10
        m = 2
        L = 169.88
        Q = 50

    with open(f"Instances/VRPSCD/t_{n}_10", "r") as file:
        customers_sets = json.load(file).values()

    with open(f"Instances/VRPSCD/realization_{n}_10", "r") as file:
        demand_scenarios = list(json.load(file).values())

    cplex_solver(customers_sets, demand_scenarios, m=int(m), L=float(L), Q=float(Q), start=int(st))

