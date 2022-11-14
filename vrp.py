import copy
import math
import random
import numpy as np
from scipy.spatial.distance import pdist, squareform
import Utils


class VRPSD(object):
    def __init__(self, env_config):

        self.vehicles = None
        self.customers = None
        self.model_type = env_config.model_type
        self.c_enc = None
        self.v_enc = None

        self.time = 0
        self.env_config = env_config
        self.ins_config = None

        self.demand_scenario = None

        # 2 heatmaps (7 by 7) are stacked over each other and there is one more line at the bottom for other info
        # self.state_shape = (2 * self.heatmap_size + 1, self.heatmap_size)
        self.n_heatmap = 4

        self.c_heatmap_dem = None
        self.c_heatmap_count = None
        self.v_heatmap_count = {}
        self.last_state = None

        self.actions = {}
        self.TT = []
        self.final_reward = 0

        self.all_distance_tables = {}
        self.distance_table = None

        self.all_heatmaps = {}

        if self.env_config.model_type == "VRPSD":
            self.demand_realization = self.demand_realization_solomon
        else:
            self.demand_realization = self.demand_realization_gendreau
        self.demand_prob = []
        self.demand_val = []

    def initialize_environment(self, instance):
        self.customers = np.array(instance["Customers"])
        self.vehicles = np.array(instance["Vehicles"])
        self.ins_config = instance["Config"]

        self.generate_heatmaps(instance["Name"])
        self.update_distance_table(instance["Name"])

    def init_encoded_env(self):
        """
        This function converts the customers and vehicles information to normalized features sets
        :return:
        """
        norm = Utils.Norms()
        n = self.ins_config.n

        #   features set for customers
        # is_realized: if the actual demand is realized, \tild{d}=\bar{d} if \hat{d}=-1 else \hat{d},
        # is_customer: indicates that the node is a customer and not a depot,
        # is_target: whether it is in the set of target customers or not
        # l_x, l_y, is_realized, \tild{d}, is_customer, is_target
        c_set = self.customers[:, [1, 2, 7, 8, 8, 8]]
        c_set[c_set[:, 0] > 0.0001, -2:] = [1., 0.]

        #   normalize them : Normalize the demand by Q
        normalizer_c = [norm.COORD, norm.COORD, 1., self.ins_config.capacity, 1., 1.]
        c_set /= normalizer_c

        # add a node as the depot
        depot = np.array([self.ins_config.depot[0] / norm.COORD,
                          self.ins_config.depot[1] / norm.COORD,
                          1., 0., -1, 0])

        # add a dummy node at the end
        dummy = np.zeros(6)

        c_set = np.vstack([c_set, depot, dummy])

        #   features set fo vehicles
        #   d_exp: expected demand to serve at its destination, loc_depot: whether it is located at the depot
        # For v: x, y, q, a, d_exp, loc_depot
        v_set = []
        for v in self.vehicles:
            exp_dem = 0
            loc_depot = 1
            l = int(v[-1])
            if l != n:
                exp_dem = self.customers[l, -1]
                loc_depot = 0
            v_set.append([v[1], v[2], v[3], v[4] - self.time, exp_dem, loc_depot])

        v_set = np.array(v_set)
        #   normalize them: divide capacity by Q
        normalizer_v = [norm.COORD, norm.COORD, self.ins_config.capacity, norm.COORD, self.ins_config.capacity, 1.]
        v_set /= normalizer_v

        return c_set, v_set

    def reset(self, instance=None, scenario=None, reset_distance=True, normalize=True):
        self.time = 0

        if instance is None:
            # don't change the customers realization
            # update availability, realized demand, unserved demand, is realized
            if normalize:
                q = self.ins_config.capacity / self.ins_config.q_normalizer
            else:
                q = self.ins_config.capacity
            self.customers[:, [3, 5, 6, 7]] = [1, -1, -1, 0]
            # update the last demand to exp dem
            self.customers[:, -1] = self.customers[:, 4]

            self.vehicles[:, 1:] = [self.ins_config.depot[0], self.ins_config.depot[1], q, 0,
                                    self.ins_config.n]

        else:
            self.customers = np.array(instance["Customers"])
            self.vehicles = np.array(instance["Vehicles"])
            self.ins_config = instance["Config"]

        if reset_distance:
            # when the set of customers might be changed, the distance tables will be changed here,
            # generate them and add to the memory if it is not generated yet, otherwise, use the memory
            self.update_distance_table(instance["Name"])

        self.demand_scenario = scenario
        # self.c_enc, self.v_enc = self.init_encoded_env()

        self.actions = {}
        for m in range(self.ins_config.m):
            self.actions[m] = [-2]
        instance_name = None if instance is None else instance["Name"]
        self.generate_heatmaps(instance_name)

        if self.model_type == "VRPSD":
            if self.ins_config.stoch_type == 0:
                self.demand_prob = [0.05, 0.9, 0.05]
                self.demand_val = [0.5, 1., 1.5]
            elif self.ins_config.stoch_type == 1:
                self.demand_prob = [0.05, 0.15, 0.6, 0.15, 0.05]
                self.demand_val = [0, 0.5, 1., 1.5, 2.]
            else:
                self.demand_prob = [0.2, 0.2, 0.2, 0.2, 0.2]
                self.demand_val = [0, 0.5, 1., 1.5, 2.]
        else:
            self.demand_prob = [0.05, 0.15, 0.6, 0.15, 0.05]
            self.demand_val = [0, 0.5, 1., 1.5, 2.]

    def update_distance_table(self, instance_name):
        if instance_name in self.all_distance_tables:
            self.distance_table = squareform(self.all_distance_tables[instance_name])
        else:
            # generate distance table
            pos_list = list(self.customers[:, 1:3])
            pos_list.append(list(self.ins_config.depot))
            distance_table = pdist(np.array(pos_list))
            self.all_distance_tables[instance_name] = distance_table
            self.distance_table = squareform(distance_table)

    def remove_from_distance_tables(self, instance_name):
        self.all_distance_tables.pop(instance_name, None)

    def generate_heatmaps(self, instance_name):
        if instance_name is None:
            h = list(self.all_heatmaps.values())[0]
            self.c_heatmap_dem = np.array(h[0])
            self.c_heatmap_count = np.array(h[1])
        elif instance_name in self.all_heatmaps:
            h = self.all_heatmaps[instance_name]
            self.c_heatmap_dem = np.array(h[0])
            self.c_heatmap_count = np.array(h[1])
        else:
            # zones
            zone_len = [self.env_config.service_area[0] / self.env_config.xy_steps[0],
                        self.env_config.service_area[1] / self.env_config.xy_steps[1]]
            zones = [
                [int(math.floor(c[1] / zone_len[1]) * self.env_config.xy_steps[1] + math.floor(c[0] / zone_len[0])),
                 c[2]] for c in self.customers[:, [1, 2, 4]]]
            c_heatmap_dem = np.zeros(self.env_config.xy_steps[0] * self.env_config.xy_steps[1])
            c_heatmap_count = np.zeros(self.env_config.xy_steps[0] * self.env_config.xy_steps[1])

            for z in zones:
                zid = int(z[0])
                if zid > 100:
                    print("asd")
                c_heatmap_count[zid] += 1. / 5.
                c_heatmap_dem[zid] += z[1]
            self.all_heatmaps[instance_name] = (c_heatmap_dem, c_heatmap_count)
            self.c_heatmap_dem = np.array(c_heatmap_dem)
            self.c_heatmap_count = np.array(c_heatmap_count)

    @staticmethod
    def get_seq_distance(seq):
        seq = np.array(seq)
        distance = 0
        for i in range(len(seq) - 1):
            distance += np.linalg.norm(seq[i] - seq[i + 1])
        return distance

    def post_decision(self, x, k, update_heatmap=True):
        # in this function, the current state transits to the post decision state.
        # it means, action x, only blocks customer x temporary to not be served by any other vehicles and
        # updates the position and arival time of the vehicle k to respectively x and get_distance(l_k, x)
        depot = self.ins_config.depot
        v_k = self.vehicles[k]
        q = v_k[3]
        loc_id = int(v_k[5])
        n = self.ins_config.n

        # depot
        if x == n:
            if loc_id == n:
                at = self.ins_config.duration_limit
            else:
                at = self.time + self.distance_table[n][loc_id]

            psi = depot
            loc_id = n
        else:
            c = self.customers[x]

            psi_x = c[1:3]
            at = self.time + self.distance_table[loc_id][x]

            psi = psi_x
            loc_id = x

            # make the customer unavailable
            c[3] = 0

            # if update_heatmap:
            # update c_heatmap_dem and count
            zone_len = [self.env_config.service_area[0] / self.env_config.xy_steps[0],
                        self.env_config.service_area[1] / self.env_config.xy_steps[1]]

            x_zone = math.floor(c[2] / zone_len[1]) * self.env_config.xy_steps[1] + math.floor(c[1] / zone_len[0])
            self.c_heatmap_count[x_zone] -= 1 / 5.

            self.c_heatmap_dem[x_zone] -= c[-1]

        # Update the V_k in Global state
        v_k[3] = q
        v_k[4] = at
        v_k[1:3] = psi
        v_k[5] = loc_id

        return at

    def state_transition(self, k, normalize=True):
        v_k = self.vehicles[k]
        n = self.ins_config.n

        served_demand = 0
        loc_id = int(v_k[5])

        if loc_id in [-2, n]:
            # restocking = 1
            # v_k[3] = 1.
            if normalize:
                v_k[3] = self.ins_config.capacity / self.ins_config.q_normalizer
            else:
                v_k[3] = self.ins_config.capacity
        else:
            # loc_id
            cur_cus = self.customers[loc_id]

            # if the demand is not realized yet, get a realization
            if cur_cus[6] == -1:
                cur_cus[5] = self.demand_scenario[loc_id]
                cur_cus[6] = self.demand_scenario[loc_id]
                cur_cus[7] = 1
                cur_cus[8] = self.demand_scenario[loc_id]

            served_demand = min(cur_cus[5], v_k[3])

            cur_cus[5] -= served_demand
            cur_cus[-1] = cur_cus[5] + 0.
            v_k[3] -= served_demand

            cur_cus[3] = cur_cus[5] > 1e-5

            if cur_cus[3]:
                # this customer will be available again with realized demand
                cur_cus[-2] = 1.
                # update the heatmap and bring back this customer to the map
                zone_len = [self.env_config.service_area[0] / self.env_config.xy_steps[0],
                            self.env_config.service_area[1] / self.env_config.xy_steps[1]]
                x_zone = int(math.floor(cur_cus[2] / zone_len[1]) * self.env_config.xy_steps[1] +
                             math.floor(cur_cus[1] / zone_len[0]))
                self.c_heatmap_dem[x_zone] += cur_cus[-1]
                self.c_heatmap_count[x_zone] += 1 / 5

        return served_demand

    def demand_realization_solomon(self, j):
        stoch_type = self.ins_config.stoch_type

        current_c = self.customers[j]

        exp_demand = current_c[4]

        if stoch_type == 2:
            op = [0, 0.5, 1.0, 1.5, 2.0]
            # op = [0.5, 1.0, 1.5, 2.0]
            rnd = random.choice(op)
            realized_demand = exp_demand * rnd

        elif stoch_type == 1:

            op = [0, 0.5, 1.0, 1.5, 2.0]
            pr = [0.05, 0.2, 0.8, 0.95, 1.0]
            # op = [0.5, 1.0, 1.5, 2.0]
            # pr = [0.158, 0.79, 0.983, 1.0]
            rnd = random.random()
            for i in range(len(pr)):
                if rnd <= pr[i]:
                    realized_demand = exp_demand * op[i]
                    break

        elif stoch_type == 0:
            op = [0.5, 1.0, 1.5]
            pr = [0.05, 0.95, 1.0]
            rnd = random.random()
            for i in range(len(pr)):
                if rnd <= pr[i]:
                    realized_demand = exp_demand * op[i]
                    break

        else:
            realized_demand = exp_demand
        # realized_demand = math.ceil(realized_demand)
        return realized_demand

    def demand_realization_gendreau(self, exp_dem):
        # c = [0, .5, 1., 1.5, 2.]
        # p = [0.05, 0.15, 0.6, 0.15, 0.05]
        # return np.random.choice(c, p=p) * exp_dem
        if exp_dem > 5 / self.ins_config.capacity:
            d = (random.randint(0, 10) - 5) / self.ins_config.capacity + exp_dem
        else:
            d = random.randint(1, 9) / self.ins_config.capacity
        return d

    def get_available_customers(self, k: int, tc_len=10):
        v_k = self.vehicles[k]
        loc_id = int(v_k[5])
        n = self.ins_config.n

        is_terminal = 0
        f_action_set = []

        # if it has no capacity go to the depot
        if v_k[3] == 0:
            # d = self.distance_table[loc_id][n]
            # f_action_set.append([-2, d, 0])
            f_action_set.append(n)
        else:
            dist_to_c = self.distance_table[loc_id, :]
            dist_to_depot = self.distance_table[n, :]
            avail_customers_cond = self.customers[:, 3] == 1
            feas_customers_cond = (dist_to_c + dist_to_depot <=
                                   (self.ins_config.duration_limit - self.time))[:-1]
            feas_customers = self.customers[np.logical_and(avail_customers_cond, feas_customers_cond), :]

            # target customers
            idxs = feas_customers[:, 0].astype(int)
            demands = feas_customers[:, -1]

            # Normalized
            demands[np.greater(demands, 1.)] = 1.
            # demands[np.greater(demands, self.ins_config.capacity)] = self.ins_config.capacity

            mrate = demands / dist_to_c[idxs]
            c_mrate = list(zip(idxs, mrate))

            sorted_c = sorted(c_mrate, reverse=True, key=lambda x: x[1])

            target_customers = sorted_c[:tc_len]
            f_action_set = [c[0] for c in target_customers]

            if loc_id != n or len(f_action_set) == 0:
                # f_action_set.append([-2, dist_to_c[n], 0])
                f_action_set.append(n)
                if loc_id == n:
                    is_terminal = 1

        return f_action_set, is_terminal

    def compute_expected_reward(self, k, cid):
        if cid == self.ins_config.n:
            return 0

        c = self.customers[cid]
        q = self.vehicles[k][3]

        if c[-2] == 1:
            return min(q, c[-1])

        kk = sum(min(q, x * c[-1]) * y for x, y in zip(self.demand_val, self.demand_prob))

        # if c[-1] == 5 / self.ins_config.capacity:
        #     v = np.array(range(1, 10)) / self.ins_config.capacity
        #     p = np.array([1. / len(v)] * len(v))
        # else:
        #     v = np.array(range(-5, 6)) / self.ins_config.capacity + c[-1]
        #     p = np.array([1. / len(v)] * len(v))
        # kk = sum(min(q, x) * y for x, y in zip(v, p))
        return kk


class VRPSimulator(object):
    def __init__(self, env: VRPSD):
        self.env = env

    def simulate(self, instance, scenario, method):
        if method not in ["random", "greedy", "normalized_greedy"]:
            raise Exception("method is not defined.")

        actions = {}

        self.env.reset(instance, normalize=False)
        self.env.demand_scenario = scenario
        expected_demand = sum(scenario)
        final_reward = 0
        # time scheduler
        TT = []
        for j in range(self.env.ins_config.m):
            TT.append((j, 0))
            actions[j] = [-2]

        final_reward = 0
        n_routes = 0
        n_served = 0
        avg_terminal_time = 0
        visited_nodes = [[[50, 50]], [[50, 50]]]
        while len(TT) > 0:
            TT.sort(key=lambda x: x[1])
            k, time = TT.pop(0)
            self.env.time = time

            r_k = self.env.state_transition(k)
            is_terminal = 0
            if self.env.vehicles[k][3] == 0:
                x_k = self.env.ins_config.n
            else:
                # get customers available around
                c_set, _ = self.env.get_available_customers(k, 20)
                if self.env.ins_config.n in c_set:
                    c_set.remove(self.env.ins_config.n)

                if len(c_set) == 0:
                    x_k = self.env.ins_config.n
                    if self.env.vehicles[k][-1] in [-2, self.env.ins_config.n]:
                        is_terminal = 1
                else:
                    # select an action for vehicle k
                    if method == "greedy":
                        x_k = c_set[0]
                    elif method == "random":
                        x_k = random.choice(c_set)
                    else:
                        return None

            if x_k in [-2, self.env.ins_config.n]:
                visited_nodes[k].append([50, 50])
            else:
                visited_nodes[k].append(self.env.customers[x_k, 1:3].tolist())

            actions[k].append(x_k)

            if x_k == self.env.ins_config.n and self.env.vehicles[k][-1] not in [-2, self.env.ins_config.n]:
                n_routes += 1
            if x_k != self.env.ins_config.n:
                n_served += 1
            if is_terminal == 1:
                avg_terminal_time += time

            # impose the action on the environment
            t_k = self.env.post_decision(x_k, k)

            # schedule the next event for vehicle k if it still has time
            if t_k < self.env.ins_config.duration_limit and is_terminal == 0:
                TT.append((k, t_k))

            final_reward += r_k
        avg_terminal_time /= self.env.ins_config.m

        # rr = VRPSD.get_seq_distance(np.array(visited_nodes[0]))
        # rr2 = VRPSD.get_seq_distance(np.array(visited_nodes[1]))
        results = Utils.TestResults(final_reward=final_reward, actions=actions, n_routes=n_routes,
                                    n_fully_served=n_served,
                                    avg_travel_time=avg_terminal_time, expected_demand=expected_demand)
        results.final_reward *= Utils.Norms.Q
        results.expected_demand *= Utils.Norms.Q

        return results

    def simulate_fixed_route(self, instance, scenario, fixed_route):
        actions = {}
        self.env.fixed_route = copy.deepcopy(fixed_route)

        self.env.reset(instance)
        self.env.demand_scenario = scenario
        expected_demand = sum(scenario)

        # time scheduler
        TT = []
        for j in range(self.env.ins_config.m):
            TT.append((j, 0))
            actions[j] = [-2]

            # the first item in each route is the depot, take it out
            self.env.fixed_route[j].pop(0)

        final_reward = 0
        n_routes = 0
        n_served = 0
        avg_terminal_time = 0
        while len(TT) > 0:
            TT.sort(key=lambda x: x[1])
            k, time = TT.pop(0)
            self.env.time = time

            is_terminal = 0
            served_demand = 0
            if self.env.vehicles[k][3] == 0:
                x_k = 0
                t_k = self.env.time + self.env.distance_table[int(self.env.vehicles[k][-1]), -1]
                self.env.vehicles[k][3] = self.env.ins_config.capacity
            else:
                if len(self.env.fixed_route[k]) > 0:
                    x_k = self.env.fixed_route[k].pop(0)
                    if x_k > 0:
                        cid = x_k - 1
                        # if self.env.demand_scenario[cid] == 0:
                        #     x_k = self.env.fixed_route[k].pop(0)
                        #     cid = x_k - 1

                        if self.env.distance_table[int(self.env.vehicles[k][-1]), cid] + self.env.distance_table[
                            cid, -1] > self.env.ins_config.duration_limit - self.env.time:
                            # not feasible
                            x_k = 0
                            # t_k = self.env.ins_config.duration_limit
                            # is_terminal = 1
                        else:
                            # feasible action for visiting a customer
                            served_demand = min(self.env.vehicles[k][3], self.env.demand_scenario[cid])
                            self.env.demand_scenario[cid] -= served_demand
                            if self.env.demand_scenario[cid] > 0:
                                self.env.fixed_route[k].insert(x_k, 0)
                            self.env.vehicles[k][3] -= served_demand

                            t_k = self.env.time + self.env.distance_table[int(self.env.vehicles[k][-1]), cid]

                    else:
                        self.env.vehicles[k][3] = self.env.ins_config.capacity
                        t_k = self.env.time + self.env.distance_table[int(self.env.vehicles[k][-1]), -1]
                else:
                    # no more action to do
                    x_k = 0
                    t_k = self.env.ins_config.duration_limit
                    is_terminal = 1

            if x_k == 0:
                self.env.vehicles[k][-1] = self.env.ins_config.n
            else:
                self.env.vehicles[k][-1] = x_k - 1

            actions[k].append(x_k)
            if x_k == 0 and self.env.vehicles[k][-1] != 0:
                n_routes += 1
            if x_k != 0:
                n_served += 1

            if is_terminal == 0:
                TT.append((k, t_k))

            final_reward += served_demand
        avg_terminal_time /= self.env.ins_config.m
        results = Utils.TestResults(final_reward=final_reward, actions=actions, n_routes=n_routes,
                                    avg_travel_time=avg_terminal_time, expected_demand=expected_demand)

        return results
