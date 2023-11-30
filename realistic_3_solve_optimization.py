import pandas as pd
from gurobipy import *
import time
import numpy as np

# for different experiments change the values of number_of_automobile and number_of_routes
number_of_automobile = 45
number_of_routes = 25

limit_reload = 10
number_of_auto_carrier = 2
number_of_sets = 800
version = "v1"
cost_per_mile = {1: 2, 2: 3}
cost_per_reload = {1: 50, 2: 100}
mile_column = {
    "Policy 0": "miles_p0",
    "Policy 1b": "miles_1b",
    "Policy 1c": "miles_1c"
}
reloads_column = {
    "Policy 0": "number_of_reloads_p0",
    "Policy 1b": "number_of_reloads_1b",
    "Policy 1c": "number_of_reloads_1c"
}

sol_time_column = {
    "Policy 0": "sol_time_p0",
    "Policy 1b": "sol_time_p1b",
    "Policy 1c": "sol_time_p1c"
}
auto_carrier_limit = {
    45: 5,
    90: 20
}


def solve_overall_problem(policy, data):

    data_table = data[~data[mile_column[policy]].isna()]
    model = Model("ColumnGeneration")
    r = model.addVars(data_table.route_index, vtype=GRB.INTEGER, ub=1, lb=0, name="var_")
    obj_fun = sum(r[route_info["route_index"]] * (
            route_info[mile_column[policy]] * route_info["cost_per_mile"] + route_info[
        reloads_column[policy]] * route_info["cost_per_reload"])
                  for i, route_info in data_table.iterrows())
    model.setObjective(obj_fun, GRB.MINIMIZE)

    routes_covered_by_am = {}
    routes_covered_by_ac = {}
    for i, route_info in data_table.iterrows():
        for am_id in route_info["automobiles_covered"]:
            if routes_covered_by_am.get(am_id) is None:
                routes_covered_by_am[am_id] = [route_info["route_index"]]
            else:
                routes_covered_by_am[am_id].append(route_info["route_index"])

        if routes_covered_by_ac.get(route_info["auto_carrier_id"]) is None:
            routes_covered_by_ac[route_info["auto_carrier_id"]] = [route_info["route_index"]]
        else:
            routes_covered_by_ac[route_info["auto_carrier_id"]].append(route_info["route_index"])

    for am_id, route_ids in routes_covered_by_am.items():
        model.addConstr(sum(r[route_id] for route_id in route_ids) == 1, name=f"{am_id}")

    for ac_id, route_ids in routes_covered_by_ac.items():
        model.addConstr(sum(r[route_id] for route_id in route_ids) <= auto_carrier_limit[number_of_automobile],
                        name=f"{ac_id}")

    model.setParam("OutputFlag", True)
    model.setParam("MIPGap", 0)
    model.optimize()

    print("model.status = ", model.status)

    if model.status == 2:
        sol = {v: r[v].X for v in r if r[v].X > 0.01}
        return True, model, sol
    else:
        return False, None, None


data = pd.read_pickle(
    f"files/realistic/{version}/t_coef_summary_{number_of_auto_carrier}_sets_{number_of_sets}_r_{number_of_routes}_a_{number_of_automobile}_1b_1c_0.pkl")
data["cost_per_mile"] = data.auto_carrier_id.map(cost_per_mile)
data["cost_per_reload"] = data.auto_carrier_id.map(cost_per_reload)
l = []
for policy in ["Policy 0", "Policy 1b", "Policy 1c"]: #"Policy 0", "Policy 1b", "Policy 1c"
    print("policy", policy)
    t1 = time.time()
    isSolved, model, sol = solve_overall_problem(policy, data)
    if isSolved:
        print("sol=", sol.keys())
        obj = model.objVal
        total_time = data[sol_time_column[policy]].sum()+ time.time()-t1
        count_ac = len(sol.keys())
        df = data[data.route_index.isin(sol.keys())]
        total_routing_cost = sum(df[mile_column[policy]]*df.cost_per_mile)
        total_loading_cost = sum(df[reloads_column[policy]] * df.cost_per_reload)

        l.append({
            "Policy": policy,
            "Count ac": count_ac,
            "total cost": np.round(model.objVal, 2),
            "routing cost": np.round(total_routing_cost, 2),
            "loading cost": np.round(total_loading_cost, 2),
            "computation time": np.round(total_time, 2)
        })
table_result = pd.DataFrame(l)
print(table_result)
