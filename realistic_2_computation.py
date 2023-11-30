import pandas as pd
import Functions as f
import time
rt_start = time.time()

# for different experiments change the values of number_of_automobile and number_of_routes
number_of_automobile = 45
number_of_routes = 10

limit_reload = 10
number_of_auto_carrier = 2
number_of_sets = 100
version = "v1"

table = pd.read_pickle(
    f"files/realistic/{version}/routes_ac_{number_of_auto_carrier}_sets_{number_of_sets}_r_{number_of_routes}_a_{number_of_automobile}.pkl")

table["routes_solved_1b"] = None

print("Solving with policy 1b")
for i, r in table.iterrows():
    print(f"iteration = {i}")
    auto_carrier_id = r["auto_carrier"]
    level = r["auto_carrier_type"]
    n_cars = r["n_cars"]
    n_slots = r["n_slots"]
    level1_slot_count = r["n_slots_level1"]
    loading_constraints_dict = r["loading_constraints_dict"]

    config = f.get_config(level, n_slots, n_cars, level1_slot_count)

    # print(f"iteration = {i}")
    filtered_routes = table.loc[i]["routes"]
    filtered_dmatrix = table.loc[i]["filtered_dmatrix"]
    mix = r["mix_sorted"]
    # print(mix)
    new_table_1b = f.get_policy1_solution_limit_nomove(filtered_routes, mix, config, limit_reload, "p1b")
    #     new_table_limit = get_policy1b_solution(filtered_routes, all_type_and_state, config, mix, limit_reload, filtered_dmatrix, loading_constraints_dict)
    # print(new_table_limit)

    table.at[i, "routes_solved_1b"] = new_table_1b

print("Solving with policy 1c")
table["routes_solved_1c"] = None
for i, r in table.iterrows():

    auto_carrier_id = r["auto_carrier"]
    level = r["auto_carrier_type"]
    n_cars = r["n_cars"]
    n_slots = r["n_slots"]
    level1_slot_count = r["n_slots_level1"]
    loading_constraints_dict = r["loading_constraints_dict"]

    config = f.get_config(level, n_slots, n_cars, level1_slot_count)

    # print(f"iteration = {i}")
    filtered_routes = r["routes_solved_1b"]
    filtered_dmatrix = table.loc[i]["filtered_dmatrix"]
    mix = r["mix_sorted"]
    # print(mix)
    new_table_1b = f.get_policy1_solution_limit_nomove(filtered_routes, mix, config, limit_reload, "p1c")
    #     new_table_limit = get_policy1b_solution(filtered_routes, all_type_and_state, config, mix, limit_reload, filtered_dmatrix, loading_constraints_dict)
    # print(new_table_limit)

    table.at[i, "routes_solved_1c"] = new_table_1b

print("Solving with policy 0")
table["routes_solved_p0"] = None
for i, r in table.iterrows():
    auto_carrier_id = r["auto_carrier"]
    level = r["auto_carrier_type"]
    n_cars = r["n_cars"]
    n_slots = r["n_slots"]
    level1_slot_count = r["n_slots_level1"]
    loading_constraints_dict = r["loading_constraints_dict"]

    config = f.get_config(level, n_slots, n_cars, level1_slot_count)

    # print(f"iteration = {i}")
    filtered_routes = r["routes_solved_1c"]  # using the previous solution table -----------------------------------
    filtered_dmatrix = r["filtered_dmatrix"]
    filtered_dmatrix_dict = filtered_dmatrix.to_dict()

    filtered_routes["routing_cost"] = [f.routing_cost(route, filtered_dmatrix_dict) for route in filtered_routes.route2]

    mix = r["mix_sorted"]
    # print(mix)
    # filtered, loading_constraints_dict, mix, n_cars, level1_slot_count
    new_table_p0 = f.get_policy0_solution(filtered_routes, mix, loading_constraints_dict, n_cars, level1_slot_count,
                                          level, "p0")
    table.at[i, "routes_solved_p0"] = new_table_p0

perc_feasible_p0_ac1 = []
perc_feasible_p0_ac2 = []
data = []
for i, r in table.iterrows():
    auto_carrier_id = r["auto_carrier"]
    auto_carrier_routing_cost = r["routing_cost"]
    auto_carrier_reloading_cost = r["reloading_cost"]
    automobiles_covered = r["automobiles"]

    solver_table = r["routes_solved_p0"]
    # auto_carrier_routing_cost = $cost /mile travelling
    solver_table["routing_cost_dollar"] = solver_table["routing_cost"] * auto_carrier_routing_cost

    # ------- policy 1b---------------------------

    # auto_carrier_reloading_cost = $cost /reload travelling
    solver_table["reloading_cost_dollar_1b"] = solver_table["cost_p1b"] * auto_carrier_reloading_cost
    solver_table["total_cost_dollar_1b"] = solver_table["routing_cost_dollar"] + solver_table[
        "reloading_cost_dollar_1b"]

    sorted_table = solver_table.sort_values(by="total_cost_dollar_1b")
    optimum_selection_1b = sorted_table.iloc[0]

    optimum_cost_1b = optimum_selection_1b["total_cost_dollar_1b"]
    number_of_reloads_1b = optimum_selection_1b["cost_p1b"]
    sol_time_p1b = sum(sorted_table.sol_time_p1b)
    # ------- policy 1b---------------------------

    # ------- policy 1c---------------------------
    solver_table["reloading_cost_dollar_1c"] = solver_table["cost_p1c"] * auto_carrier_reloading_cost
    solver_table["total_cost_dollar_1c"] = solver_table["routing_cost_dollar"] + solver_table[
        "reloading_cost_dollar_1c"]

    sorted_table = solver_table.sort_values(by="total_cost_dollar_1c")
    optimum_selection_1c = sorted_table.iloc[0]

    optimum_cost_1c = optimum_selection_1c["total_cost_dollar_1c"]
    number_of_reloads_1c = optimum_selection_1c["cost_p1c"]
    sol_time_p1c = sum(sorted_table.sol_time_p1c)

    # ------- policy 1c---------------------------

    # ------- policy 0---------------------------

    table_feasible_p0 = solver_table[solver_table.isLifo == True]
    #     print(i, len(table_feasible_p0))

    if auto_carrier_id == 1:
        perc_feasible_p0_ac1.append(len(table_feasible_p0) * 100 / len(solver_table))

    else:
        perc_feasible_p0_ac2.append(len(table_feasible_p0) * 100 / len(solver_table))

    sorted_table_feasible_p0 = table_feasible_p0.sort_values(by="routing_cost_dollar")

    if len(sorted_table_feasible_p0) == 0:
        print(f"id = {i}")
    optimum_selection_p0 = None
    cost_p0 = None
    sol_time_p0 = None
    miles_p0 = None

    if len(table_feasible_p0) > 0:
        optimum_selection_p0 = sorted_table_feasible_p0.iloc[0]
        cost_p0 = optimum_selection_p0["routing_cost_dollar"]
        sol_time_p0 = sum(sorted_table_feasible_p0.sol_time_p0)
        miles_p0 = optimum_selection_p0["routing_cost"]

    # ------- policy 0---------------------------

    data.append({
        "route_index": i,
        "auto_carrier_id": auto_carrier_id,
        "automobiles_covered": automobiles_covered,

        "optimum_cost_1b": optimum_cost_1b,
        "number_of_reloads_1b": number_of_reloads_1b,
        "sol_time_p1b": sol_time_p1b,
        "optimum_selection_1b": optimum_selection_1b,
        "miles_1b": optimum_selection_1b["routing_cost"],

        "optimum_cost_1c": optimum_cost_1c,
        "number_of_reloads_1c": number_of_reloads_1c,
        "sol_time_p1c": sol_time_p1c,
        "optimum_selection_1c": optimum_selection_1c,
        "miles_1c": optimum_selection_1c["routing_cost"],

        "cost_p0": cost_p0,
        "miles_p0": miles_p0,
        "sol_time_p0": sol_time_p0,
        "optimum_selection_p0": optimum_selection_p0,
        "number_of_reloads_p0": 0,
        "filtered_dmatrix": r["filtered_dmatrix"]
    })


data_table = pd.DataFrame(data)
data_table.to_pickle(f"files/realistic/{version}/t_coef_summary_{number_of_auto_carrier}_sets_{number_of_sets}_r_{number_of_routes}_a_{number_of_automobile}_1b_1c_0.pkl")
print("runtime = ", (time.time()-rt_start))