import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns
import numpy as np
import time
from itertools import permutations
from itertools import combinations
import itertools
import networkx as nx
import copy
from datetime import datetime
from sortedcontainers import SortedList


def get_all_states(n_slot, n_car, v_type, loading_constraints_dict):
    all_state_options = {}
    cars = list(range(1, n_car + 1))

    for n_selected_car in range(n_car + 1):
        slot_indexes = list(permutations(list(range(n_slot)), n_selected_car))
        for car_ids in combinations(cars, n_selected_car):
            key = tuple(sorted(car_ids))
            values = []
            # print("car_ids", car_ids)
            for slot_index in slot_indexes:
                zeros = np.zeros(n_slot, dtype=int)
                for (index, replacement) in zip(slot_index, car_ids):
                    zeros[index] = replacement
                # print("zeros", zeros)
                if is_state_feasible2(zeros, loading_constraints_dict, v_type):
                    values.append(tuple(zeros))
            all_state_options[key] = values
    return all_state_options


def get_all_type_and_valid_state_type_specific(n_slot, loading_constraints_dict, v_type_list):
    all_type_and_state = {}
    for index, v_type in enumerate(v_type_list):
        print(index, v_type)
        states = get_all_states(n_slot, n_car=n_slot, v_type=v_type, loading_constraints_dict=loading_constraints_dict)

        all_type_and_state[tuple(v_type)] = states
    return all_type_and_state


def generate_routes(vids):
    route = []
    up = set()
    down = set(vids)

    while (len(down) > 0 or len(up)) > 0:
        action = random.choice([1, -1])
        if action == 1 and len(down) > 0:
            car = random.choice(list(down))
            up.add(car)
            down.discard(car)
            route.append((action, car))
        elif action == -1 and len(up) > 0:
            car = random.choice(list(up))
            up.discard(car)
            route.append((action, car))
    return route


def get_config(level, n_slot, n_cars, level1_slot_count):
    constraints = pd.read_csv(f"files/given/constraints/{level}/loading_constraint_{n_slot}_car.csv", index_col="index")
    constraints = constraints.fillna(0)
    constraints = constraints.astype(int)

    config = {
        "n_slot": n_slot,
        "n_cars": n_cars,
        "loading_constraints": constraints,
        "loading_constraints_dict": constraints.to_dict(),
        "level": level,
        "level1_slot_count": level1_slot_count
    }
    return config


def get_policy1_solution_limit_nomove(new_table, v_type, config, link_cost_limit, pol):
    loading_constraints_dict = config["loading_constraints_dict"]

    new_table[f"cost_{pol}"] = None
    new_table[f"sol_time_{pol}"] = None
    new_table[f"path_{pol}"] = None
    new_table[f"reload_count_{pol}"] = None
    new_table[f"graph_{pol}"] = None
    new_table[f"graph_info_{pol}"] = None
    for i, r in new_table.iterrows():
        # print(i)
        route = r["route2"]

        start_time = time.time()
        path, cost, err, reload_count, graph = None, None, None, None, None

        if pol == "p1b":
            path, cost, err, reload_count, graph, stage_nodes = generate_ssn_solve_p1b_labeling_algo_full(route, config,
                                                                                                          v_type,
                                                                                                          link_cost_limit,
                                                                                                          loading_constraints_dict)

        if pol == "p1c":
            path, cost, err, reload_count, graph = generate_ssn_solve_p1c(route, config, v_type, link_cost_limit,
                                                                          loading_constraints_dict)

        runtime = time.time() - start_time
        new_table.at[i, f"sol_time_{pol}"] = runtime
        new_table.at[i, f"path_{pol}"] = path
        new_table.at[i, f"graph_{pol}"] = None
        new_table.at[i, f"cost_{pol}"] = cost
        new_table.at[i, f"reload_count_{pol}"] = tuple(reload_count)
        if graph != None:
            new_table.at[i, f"graph_info_{pol}"] = len(graph.nodes()), len(graph.edges())

    return new_table


def generate_ssn_solve_p1b_labeling_algo_full(route, config, v_type, link_cost_limit, loading_constraints_dict,
                                              draw=False):
    G = nx.DiGraph()

    level = config["level"]
    n_slot = config["n_slot"]
    n_cars = config["n_cars"]
    level1_slot_count = config["level1_slot_count"]
    start_node = tuple([0] * n_slot)
    end_node = tuple([0] * n_slot)
    error_message = ""
    stage_nodes = {0: {tuple([0] * n_slot): 0}}
    prev_labels = stage_nodes[0]

    for index, (action, vid) in enumerate(route):
        current_labels = {}
        # print(f"prev_labels = {prev_labels}")
        for state_pre in prev_labels.keys():
            new_state_list = []
            if action == 1:
                new_state_list = set(get_states_adding_new_vids(state_pre, vid))
            else:
                if level == "single_level":
                    new_state_list = set(
                        get_states_removing_vids_partial_intra_slot_movement_single_level(state_pre, vid))

                if level == "bilevel":
                    new_state_list = set(
                        get_states_removing_vids_partial_intra_slot_movement_bilevel(state_pre, vid, level1_slot_count))

            for state_current_new in new_state_list:
                if is_state_feasible2(state_current_new, loading_constraints_dict, v_type):
                    current_labels[state_current_new] = 1000
        if len(current_labels.keys()) == 0:
            return None, None, "current_state_list len 0", [], None, None

        for state_current_new, label_current in current_labels.items():
            for state_prev, lavel_prev in prev_labels.items():
                # print(state_prev, state_current_new)
                if lavel_prev < label_current:
                    count_reload = get_count_reload(state_prev, state_current_new, level, level1_slot_count)
                    if count_reload <= link_cost_limit:
                        if count_reload + lavel_prev < label_current:
                            label_current = count_reload + lavel_prev
                            current_labels[state_current_new] = label_current
                            # print(state_prev, state_current_new, count_reload)
                            current_state_name = str(state_current_new) + "-" + str(index + 1)
                            # print()
                            G.add_edge(str(state_prev) + "-" + str(index), current_state_name, length=count_reload)
                else:
                    break
        #             print("--------------------------------------")

        current_labels = {k: v for k, v in sorted(current_labels.items(), key=lambda item: item[1])}
        stage_nodes[index + 1] = current_labels
        prev_labels = current_labels

    shortest_path = None
    shortest_path_length = None
    reload_count = []

    if nx.has_path(G, str(start_node) + "-0", str(start_node) + "-" + str(n_cars * 2)):
        shortest_path = nx.shortest_path(G, str(start_node) + "-0", str(start_node) + "-" + str(n_cars * 2),
                                         weight='length')

        pathGraph = nx.path_graph(shortest_path)
        for ea in pathGraph.edges():
            reload_count.append(G.edges[ea[0], ea[1]]["length"])

        shortest_path_length = sum(reload_count)

    #     n_reloads_1b2 = None
    #     if stage_nodes!= None:
    #         if n_slot*2 in stage_nodes.keys():
    #             n_reloads_1b2 = stage_nodes[n_slot*2][end_node]

    return shortest_path, shortest_path_length, "", reload_count, G, stage_nodes


def generate_ssn_solve_p1c(route, config, v_type, link_cost_limit, loading_constraints_dict, draw=False):
    """
    p1c: generate state from previous state, no reshuffle during load and unload, connect only the induced states between
    two stages
    """
    A = nx.DiGraph()
    level = config["level"]
    n_slot = config["n_slot"]
    level1_slot_count = config["level1_slot_count"]
    prev_state_list = [tuple([0] * n_slot)]
    A.add_node(str(prev_state_list[0]) + "-" + str(0), pos=(0, 0))
    error_message = ""
    for index, (action, vid) in enumerate(route):

        current_valid_state_list = set()

        # print(f"prev_state_list = {prev_state_list}")

        for state_pre in prev_state_list:
            new_state_list = []
            if action == 1:
                new_state_list = set(get_states_adding_new_vids(state_pre, vid))
            else:
                new_state_list = set(get_states_removing_vids(state_pre, vid))

            for state_current_new in new_state_list:
                # print(f"state_current_new = {state_current_new}")
                if is_state_feasible2(state_current_new, loading_constraints_dict, v_type):
                    # print(f"state_current_new = {state_current_new}")
                    count_reload = get_count_reload(state_pre, state_current_new, level, level1_slot_count)
                    # print(f"{level}, {state_pre}, {state_current}, level1_slot_count = {level1_slot_count} reload ={count_reload}")
                    if count_reload <= link_cost_limit:
                        current_state_name = str(state_current_new) + "-" + str(index + 1)
                        current_valid_state_list.add(state_current_new)
                        A.add_edge(str(state_pre) + "-" + str(index), current_state_name, length=count_reload)

        prev_state_list = list(current_valid_state_list)

        if len(prev_state_list) == 0:
            return None, None, None, "current_state_list len 0", None

    start_node, end_node = f"{str(tuple([0] * n_slot))}-0", f"{str(tuple([0] * n_slot))}-" + str(len(route))

    shortest_path = None
    shortest_path_length = None
    reload_count = []
    if nx.has_path(A, start_node, end_node):
        shortest_path = nx.shortest_path(A, start_node, end_node, weight='length')
        pathGraph = nx.path_graph(shortest_path)
        shortest_path_length = sum([A.edges[e[0], e[1]]["length"] for e in pathGraph.edges(data=True)])
        # print(shortest_path)

        # print(A.nodes(data = True))
        # print(route)

        edge_labels = {}
        for ea in pathGraph.edges():
            # print from_node, to_node, edge's attributes
            # print(ea, A.edges[ea[0], ea[1]]["length"])
            edge_labels[ea] = A.edges[ea[0], ea[1]]["length"]
            reload_count.append(A.edges[ea[0], ea[1]]["length"])


    else:
        error_message = "No path found"
    return shortest_path, shortest_path_length, error_message, reload_count, A


def get_states_adding_new_vids(state, vid):
    state = list(state)
    for index in range(len(state)):
        new_state = state[:]
        if new_state[index] == 0:
            new_state[index] = vid
            yield tuple(new_state)


def get_states_removing_vids(state, vid):
    state = list(state)
    for index in range(len(state)):
        new_state = state[:]
        if new_state[index] == vid:
            new_state[index] = 0
            yield tuple(new_state)


from itertools import permutations


def get_states_adding_new_vids_partial_intra_slot_movement_single_level(state, vid):
    state = list(state)
    for index in range(len(state)):
        if state[index] == 0:
            new_state = state[:]
            new_state[index] = vid
            shuffle_items = new_state[:index]
            fixed_items = new_state[index:]
            for each_shuffle_item in permutations(shuffle_items):
                yield tuple(list(each_shuffle_item) + fixed_items)


def get_states_removing_vids_partial_intra_slot_movement_single_level(state, vid):
    state = list(state)
    for index in range(len(state)):
        if state[index] == vid:
            new_state = state[:]
            new_state[index] = 0
            shuffle_items = new_state[:index + 1]
            fixed_items = new_state[index + 1:]
            for each_shuffle_item in permutations(shuffle_items):
                yield tuple(list(each_shuffle_item) + fixed_items)


def get_states_removing_vids_partial_intra_slot_movement_bilevel(state, vid, l1_n_slot):
    state = list(state)
    l1 = state[:l1_n_slot]
    l2 = state[l1_n_slot:]

    #     print(f"l1 = {l1}, l2 = {l2}")

    shuffle_index = set()
    shuffle_vid = set()

    if vid in l1:
        index = l1.index(vid)
        #         print(np.array(range(index+1)))
        shuffle_index.update(np.array(range(index + 1)))
        shuffle_vid.update(l1[:index])
        for i in range(len(l2)):
            if l2[i] == 0:
                shuffle_index.add(i + l1_n_slot)
            else:
                break

    if vid in l2:
        index = l2.index(vid)
        shuffle_index.update(np.array(range(index + 1)) + l1_n_slot)
        shuffle_index.add(0)
        shuffle_vid.add(l1[0])
        shuffle_vid.update(l2[:index])
        for i, v in enumerate(l1[1:]):
            if v == 0:
                shuffle_index.add(i + 1)
            else:
                break

    shuffle_vid.discard(0)
    shuffle_vid = list(shuffle_vid)

    state[state.index(vid)] = 0
    for i in shuffle_index:
        state[i] = 0

    for p in permutations(shuffle_index, len(shuffle_vid)):
        new_state = state[:]
        for i, idx in enumerate(p):
            new_state[idx] = shuffle_vid[i]
        yield (tuple(new_state))


def is_state_feasible2(state, loading_constraints_dict, v_type):
    filtered_index = [str(i) + v_type[vid] for i, vid in enumerate(state)]
    s = 0
    for k in filtered_index:
        s = s + loading_constraints_dict[k][k]

    return s == 0.0


def get_count_reload(f_value, t_value, level, level1_slot_count):
    if level == "single_level":
        return get_count_reload_single_level(f_value, t_value)
    if level == "bilevel":
        return get_count_reload_bilevel(f_value, t_value, level1_slot_count)

    return None


def get_count_reload_single_level(f_value, t_value):
    f_value = [x for x in list(reversed(f_value)) if x > 0]
    t_value = [x for x in list(reversed(t_value)) if x > 0]
    isChange = False
    count_reload = 0
    min_len = min(len(f_value), len(t_value))
    # print(min_len)
    # print(f_value, t_value)
    for i in range(min_len):
        if f_value[i] != t_value[i]:
            isChange = True

            f1 = set(f_value[i:])
            t1 = set(t_value[i:])
            # print(f"f1 = {f1}, t1 = {t1}")

            count_reload = len(f1.intersection(t1))
            break
    return count_reload


def get_count_reload_bilevel(f_value, t_value, l1_n):
    l2_n = len(f_value) - l1_n
    f_value = list(f_value)
    t_value = list(t_value)

    froml1_to_l2 = set(f_value[:l1_n]) & set(t_value[l1_n:])
    froml2_to_l1 = set(t_value[:l1_n]) & set(f_value[l1_n:])

    level_changed = (froml1_to_l2 | froml2_to_l1) - set([0])

    f_state_l1 = [x for x in list(reversed(f_value[:l1_n])) if x > 0]
    f_state_l2 = [x for x in list(reversed([f_value[0]] + f_value[l1_n:])) if x > 0]
    t_state_l1 = [x for x in list(reversed(t_value[:l1_n])) if x > 0]
    t_state_l2 = [x for x in list(reversed([t_value[0]] + t_value[l1_n:])) if x > 0]

    f_state_l1 = f_state_l1 + [0] * (l1_n - len(f_state_l1))
    f_state_l2 = f_state_l2 + [0] * (len(f_value) - l1_n - len(f_state_l2) + 1)

    t_state_l1 = t_state_l1 + [0] * (l1_n - len(t_state_l1))
    t_state_l2 = t_state_l2 + [0] * (len(t_value) - l1_n - len(t_state_l2) + 1)

    f = set()
    t = set()

    isL1Change = False
    count_reloadL1 = 0
    for i in range(l1_n):
        if f_state_l1[i] != t_state_l1[i] or f_state_l1[i] in level_changed or t_state_l1[i] in level_changed:
            isL1Change = True
            f1 = set(f_state_l1[i:])
            t1 = set(t_state_l1[i:])

            f = f.union(f1)
            t = t.union(t1)
            break

    isL2Change = False
    count_reloadL2 = 0

    for i in range(l2_n):
        if f_state_l2[i] != t_state_l2[i] or f_state_l2[i] in level_changed or t_state_l2[i] in level_changed:
            isL2Change = True
            f2 = set(f_state_l2[i:])
            t2 = set(t_state_l2[i:])

            f = f.union(f2)
            t = t.union(t2)
            break
    f = f - set([0])
    t = t - set([0])

    reloaded = f & t
    count_reload = len(reloaded)
    return count_reload


def routing_cost(route, dist_mat_dict):
    routing_cost = 0
    route = [(1, 0)] + route + [(-1, 0)]
    route = [a * b for a, b in route]
    # print(route)
    for i in range(len(route) - 1):
        # print(i, route[i])
        f = route[i]
        t = route[i + 1]
        routing_cost = routing_cost + dist_mat_dict[f][t]
    return routing_cost


def get_policy0_solution(new_table, mix, loading_constraints_dict, n_cars, level1_slot_count, level, pol):
    new_table[f"cost_{pol}"] = None
    new_table[f"sol_time_{pol}"] = None
    new_table[f"path_{pol}"] = None
    new_table[f"isLifo"] = False

    for i, r in new_table.iterrows():
        # print(i)
        v_type = mix
        route = r["route2"]

        start_time = time.time()
        isLifo = False
        if level == "single_level":
            isLifo = simple_check_lifo_route_single_level(route, loading_constraints_dict, v_type, n_cars)
        else:
            isLifo = simple_check_lifo_route_bilevel(route, loading_constraints_dict, v_type, n_cars, level1_slot_count)
        new_table.at[i, f"isLifo"] = isLifo
        # new_table.at[i, "isRouteLifo"] = isRouteLifo(route)
        if isLifo == False:
            runtime = time.time() - start_time
            new_table.at[i, f"sol_time_{pol}"] = runtime
        else:
            runtime = time.time() - start_time
            new_table.at[i, f"sol_time_{pol}"] = runtime
            new_table.at[i, f"cost_{pol}"] = 0
    return new_table


def simple_check_lifo_route_single_level(route, loading_constraints_dict, v_type, n_slot):
    state = [0] * n_slot
    for action, vid in route:
        # print(action, "   ", vid)
        feasible = False
        first_zero_index = get_zero_index(state)
        # print(f"first_zero_index = {first_zero_index}")

        if action == 1:
            if first_zero_index is None:
                return False

            for i in list(range(first_zero_index + 1))[::-1]:
                state_check = copy.deepcopy(state)
                state_check[i] = vid

                if is_state_feasible2(state_check, loading_constraints_dict, v_type):
                    state = state_check
                    break
                elif i == 0 and is_state_feasible2(state_check, loading_constraints_dict, v_type) == False:
                    return False
        elif action == -1:
            first_nonzero_index = get_zero_index(state)

            if first_nonzero_index == None:
                first_nonzero_index = 0
            else:
                first_nonzero_index = first_nonzero_index + 1
            # print(f"action ==-1 first_nonzero_index = {first_nonzero_index}")
            if first_nonzero_index >= n_slot or state[first_nonzero_index] != vid:
                return False
            else:
                state[first_nonzero_index] = 0
        # print(state)
    return True


def simple_check_lifo_route_bilevel(route, loading_constraints_dict, v_type, n_slot, l1_n):
    d = {0: [tuple(np.zeros(n_slot, dtype=int))]}
    count = 1
    for action, vid in route:
        state_list = d[count - 1]
        # print(f"state_list = {state_list}")
        new_state_list_set = set()
        for state in state_list:
            new_states = make_children(state, action, vid, l1_n, loading_constraints_dict, v_type)
            for each_new_state in new_states:
                new_state_list_set.add(tuple(each_new_state))
        # print(f"new_state_list = {new_state_list}")
        d[count] = new_state_list_set
        count = count + 1

    return len(d[n_slot * 2]) > 0


def make_children(state, action, vid, l1_n, loading_constraints_dict, v_type):
    state = list(state)

    all_children = []
    # print(f"state = {state}")
    l1 = state[:l1_n]
    l2 = state[l1_n:]
    # print(f"state = {state}, l1 = {l1}, l2 = {l2}")
    first_zero_index_l1 = get_zero_index(l1)
    first_zero_index_l2 = get_zero_index(l2)

    # print(f"first_zero_index_l1 = {first_zero_index_l1}, first_zero_index_l2 = {first_zero_index_l2}")

    if action == 1:

        if not first_zero_index_l1 is None:
            for i in list(range(first_zero_index_l1 + 1))[::-1]:
                state_check = copy.deepcopy(state)
                # print(f"state_check 1 = {state_check}")
                state_check[i] = vid

                if is_state_feasible2(state_check, loading_constraints_dict, v_type):
                    all_children.append(state_check)
                    break

        if not first_zero_index_l2 is None and state[0] == 0:
            for i in list(range(first_zero_index_l2 + 1))[::-1]:
                state_check = copy.deepcopy(state)
                # print(f"state_check 2 = {state_check}")
                state_check[l1_n + i] = vid

                if is_state_feasible2(state_check, loading_constraints_dict, v_type):
                    all_children.append(state_check)
                    break
    else:
        first_nonzero_index1 = 0
        first_nonzero_index2 = 0

        if first_zero_index_l1 == None:
            first_nonzero_index1 = 0
        else:
            first_nonzero_index1 = first_zero_index_l1 + 1

        if first_zero_index_l2 == None:
            first_nonzero_index2 = 0
        else:
            first_nonzero_index2 = first_zero_index_l2 + 1

        # print(f"action ==-1 first_nonzero_index = {first_nonzero_index}")

        new_state = copy.deepcopy(state)

        if first_nonzero_index1 < len(state) and new_state[first_nonzero_index1] == vid:
            new_state[first_nonzero_index1] = 0
            all_children.append(new_state)

        new_state = copy.deepcopy(state)

        if l1_n + first_nonzero_index2 < len(state) and new_state[l1_n + first_nonzero_index2] == vid and state[0] == 0:
            new_state[l1_n + first_nonzero_index2] = 0
            all_children.append(new_state)
    # print("all_children")
    # print(all_children)
    return all_children


def get_zero_index(state):
    first_zero_index = len(state) - 1
    nz_indexes = np.nonzero(state)[0]
    if len(nz_indexes) != 0:
        first_zero_index = nz_indexes[0] - 1
    if first_zero_index < 0:
        first_zero_index = None
    return first_zero_index


def get_policy1_solution(filtered, all_type_and_state, config, mix, pol):
    n_cars = config["n_slot"]
    level = config["level"]

    filtered[f"sol_time_{pol}"] = None
    filtered[f"cost_{pol}"] = None
    filtered[f"path_{pol}"] = None
    filtered[f"graph_{pol}"] = None
    filtered[f"reload_count_{pol}"] = None
    filtered[f"graph_info_{pol}"] = None

    for i, r in filtered.iterrows():
        print(i)
        d = r.to_dict()
        v_type = mix
        route = r["route2"]
        start_time = time.time()
        path, cost, reload_counts, graph = generate_ssn_solve_p1a(route, all_type_and_state[tuple(v_type)], config)
        runtime = time.time() - start_time
        filtered.at[i, f"sol_time_{pol}"] = runtime
        filtered.at[i, f"cost_{pol}"] = cost
        filtered.at[i, f"path_{pol}"] = path
        filtered.at[i, f"graph_{pol}"] = None
        filtered.at[i, f"reload_count_{pol}"] = reload_counts
        filtered.at[i, f"graph_info_{pol}"] = len(graph.nodes()), len(graph.edges())

    return filtered


def generate_ssn_solve_p1a(route, state_options, config):
    """
    Policy 1a: all feasible states are pregenerated in the 'state_option'. Connect all states between stages.
    """

    A = nx.DiGraph()
    vehicles_on = SortedList()
    prev_state_list = state_options[tuple()]
    level = config["level"]
    n_slot = config["n_slot"]
    level1_slot_count = config["level1_slot_count"]

    for index, (action, vid) in enumerate(route):
        # print(index, (action, vid))
        if action == 1:
            vehicles_on.add(vid)

        else:
            vehicles_on.discard(vid)
        # print(vehicles_on)
        car_in = tuple(vehicles_on)
        current_state_list = state_options.get(car_in, [])

        if len(current_state_list) == 0:
            return None, None, None, None

        # print(f"Number state available for {car_in} is: {len(current_state_list)}")

        for state_pre in prev_state_list:
            for state_current in current_state_list:
                count_reload = get_count_reload(state_pre, state_current, level, level1_slot_count)

                A.add_edge(str(state_pre) + "-" + str(index), str(state_current) + "-" + str(index + 1),
                           length=count_reload)

        prev_state_list = current_state_list[:]
    start_node, end_node = f"{str(tuple([0] * n_slot))}-0", f"{str(tuple([0] * n_slot))}-" + str(len(route))

    shortest_path = None
    shortest_path_length = None
    reload_count = []
    if nx.has_path(A, start_node, end_node):
        shortest_path = nx.shortest_path(A, start_node, end_node, weight='length')

        pathGraph = nx.path_graph(shortest_path)
        for ea in pathGraph.edges():
            reload_count.append(A.edges[ea[0], ea[1]]["length"])

        shortest_path_length = sum([A.edges[e[0], e[1]]["length"] for e in pathGraph.edges(data=True)])
    return shortest_path, shortest_path_length, reload_count, A


def get_routing_cost(t, filtered):
    table = t[:]

    filtered_dict = filtered.to_dict()
    list_routing_cost = []
    for i, r in table.iterrows():
        route = r['route2']
        rCost = routing_cost(route, filtered_dict)
        list_routing_cost.append(rCost)
    table["routing_cost"] = list_routing_cost
    return table
