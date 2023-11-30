from sortedcontainers import SortedList
import pandas as pd
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import margo_loader
import Functions as f
import copy

prob_type = "benchmark" # prob_type = "benchmark" "from_fl", "to_fl", "fl"
mix_type = "same"   # mix_type = "same" or "variable" or "defined"
level = "bilevel" # level = "single_level" or "bilevel"
n_cars = 6
level1_slot_count = 3
number_of_routes = 250
number_of_iteration = 100
coef = 50


config = f.get_config(level, n_cars,n_cars, level1_slot_count)
loading_constraints = config["loading_constraints"]
loading_constraints_dict = loading_constraints.to_dict()

fn_experiments = f"files/benchmark_data/{number_of_iteration}_exp_{number_of_routes}_routes_{n_cars}_slots_{prob_type}_{mix_type}.pkl"
fn_all_states = f'files/all_type_and_state/{level}/all_type_and_state_{number_of_iteration}_exp_{number_of_routes}_routes_{n_cars}_slots_{prob_type}_{mix_type}.pickle'
all_type_and_state = pd.read_pickle(fn_all_states)


def get_both_policy_table_limit_link_cost(table, loading_cost_coef):
    table["loading_cost_weighted_p0"] = table["cost_p0"] * loading_cost_coef
    table["total_cost_p0"] = table["loading_cost_weighted_p0"] + table["routing_cost"]

    table["loading_cost_weighted_p1a"] = table["cost_p1a"] * loading_cost_coef
    table["loading_cost_weighted_p1b"] = table["cost_p1b"] * loading_cost_coef
    table["loading_cost_weighted_p1c"] = table["cost_p1c"] * loading_cost_coef

    table["total_cost_p1a"] = table["loading_cost_weighted_p1a"] + table["routing_cost"]
    table["total_cost_p1b"] = table["loading_cost_weighted_p1b"] + table["routing_cost"]
    table["total_cost_p1c"] = table["loading_cost_weighted_p1c"] + table["routing_cost"]
    t0 = table.copy(deep=True)
    t0 = t0[(t0.cost_p0 == 0)]
    t1 = table.copy(deep=True)
    return t0, t1

table  =  pd.read_pickle(fn_experiments)
coef_result = []
for i in range(number_of_iteration):
    print(f"iteration = {i}")
    filtered_routes = table.loc[i]["routes"]
    mix = table.loc[i]["mix_sorted"]
    filtered_routes["mix_sorted"] = str(mix)
    filtered_routes["n_cars"] = n_cars
    filtered_routes["level"] = level
    # print(filtered_routes)
    new_table = f.get_policy1_solution(filtered_routes, all_type_and_state, config, mix, "p1a")
    print(f"new_table col = {new_table.columns}")
    new_table1 = f.get_policy1_solution_limit_nomove(new_table, mix, config, 10, "p1b")
    print(f"new_table1 = {new_table1.columns}")
    new_table1 = f.get_policy1_solution_limit_nomove(new_table1, mix, config, 10, "p1c")
    print(f"new_table1 = {new_table1.columns}")

    new_table1 = f.get_policy0_solution(new_table1, mix, loading_constraints_dict, n_cars, level1_slot_count,
                                        config["level"], "p0")
    print(f"new_table1 = {new_table1.columns}")

    filtered_dmatrix, selected_pickup, selected_drop = table.loc[i]["filtered_dmatrix"], table.loc[i][
        "pickup_locations"], table.loc[i]["dropoff_locations"]

    new_table6 = f.get_routing_cost(new_table1[:], filtered_dmatrix)

    print(f"new table 6 length  = {len(new_table6)}")
    print(f"new table 6 col  = {new_table6.columns}")

    t0, t1 = get_both_policy_table_limit_link_cost(new_table6[:], coef)
    # print(t1_all.columns)
    min_total_cost_0 = None
    min_total_cost_1a = None
    min_total_cost_1b = None
    min_total_cost_1c = None
    min_total_cost_1b2 = None

    time0 = None
    time1a = None
    time1b = None
    time1c = None
    time1b2 = None

    t0_opt = None
    t1a_opt = None
    t1b_opt = None
    t1c_opt = None
    t1b2_opt = None

    if len(t0) > 0:
        t0_opt = t0.sort_values("total_cost_p0").iloc[0]
        min_total_cost_0 = t0.total_cost_p0.min()
        time0 = t0.sol_time_p0.sum()
    if len(t1) > 0:
        t1a_opt = t1.sort_values("total_cost_p1a").iloc[0]
        min_total_cost_1a = t1.total_cost_p1a.min()
        time1a = t1.sol_time_p1a.sum()

    if len(t1) > 0:
        t1b_opt = t1.sort_values("total_cost_p1b").iloc[0]
        min_total_cost_1b = t1.total_cost_p1b.min()
        time1b = t1.sol_time_p1b.sum()

    if len(t1) > 0:
        t1c_opt = t1.sort_values("total_cost_p1c").iloc[0]
        min_total_cost_1c = t1.total_cost_p1c.min()
        time1c = t1.sol_time_p1c.sum()


    data_to_add = {
        "iteration": i,
        "coef": coef,
        "mix": mix,
        "min_total_cost_0": min_total_cost_0,
        "min_total_cost_1a": min_total_cost_1a,
        "min_total_cost_1b": min_total_cost_1b,
        "min_total_cost_1c": min_total_cost_1c,
        "min_total_cost_1b2": min_total_cost_1b2,
        "time0": time0,
        'time1a': time1a,
        "time1b": time1b,
        "time1c": time1c,
        "time1b2": time1b2,
        "t0_opt": t0_opt,
        "t1a_opt": t1a_opt,
        "t1b_opt": t1b_opt,
        "t1c_opt": t1c_opt,
        "t1b2_opt": t1b2_opt,
        "t0": t0,
        "t1": t1,
        "selected_pick_loc": selected_pickup,
        "selected_drop_loc": selected_drop,
        #         "table":new_table6
    }
    # print(f"data_to_add = {data_to_add}")
    coef_result.append(data_to_add)

t_coef = pd.DataFrame(coef_result)

t_coef.to_pickle(f"files/solution/t_coef_{number_of_iteration}_exp_{number_of_routes}_routes_{n_cars}_slots_{prob_type}_{mix_type}.pkl")