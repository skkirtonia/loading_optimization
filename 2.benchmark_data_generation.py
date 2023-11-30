import pandas as pd
import numpy as np
import random
import pickle
import Functions as f

prob_type = "benchmark"  # prob_type = "benchmark" "from_tampa", "to_tampa"
mix_type = "same"  # mix_type = "same" or "variable" or "defined"
level = "bilevel"  # level = "single_level" or "bilevel"
n_cars = 6
n_slot = 6
level1_slot_count = 3
number_of_routes = 250
number_of_iteration = 100

expected_mix = ["T1", "T2", "T2", "T3", "T3", "T3"]

config = f.get_config(level, n_slot, n_cars, level1_slot_count)
loading_constraints = config["loading_constraints"]
loading_constraints_dict = loading_constraints.to_dict()

origin_dest_table = None
if prob_type == "benchmark":
    origin_dest_table = pd.read_pickle("files/real_data/order_pick_drop_distance_real.pkl")
if prob_type == "from_tampa":
    origin_dest_table = pd.read_pickle("files/real_data/order_pick_from_tampa_drop_distance_real.pkl")

if prob_type == "to_tampa":
    origin_dest_table = pd.read_pickle("files/real_data/order_pick_drop_to_tampa_distance_real.pkl")

dMatrix = pd.read_pickle('files/real_data/distance_matrix_fl_road.pkl')

def get_filtered_dMatrix_real(n_od_pair_table, dMatrix, n_cars):
    n_od_pair_table1 = n_od_pair_table.sample(n_cars)

    depot = [0]
    pickup = list(n_od_pair_table1.origin)
    drop = list(n_od_pair_table1.destination)

    # print(f"vids = {vids}, depot={depot}, pick = {pickup}, drop = {drop}")
    rand_ids = depot + pickup + drop
    # print(rand_ids)
    filtered = dMatrix.loc[rand_ids, rand_ids]
    filtered = filtered.reset_index(drop=True)

    ids = list(range(1, n_cars + 1))
    index_col = [0] + ids + list(np.array(ids) * -1)
    filtered = filtered.set_index([pd.Index(index_col)])
    filtered.columns = index_col
    return filtered, pickup, drop


data = []
for i in range(number_of_iteration):
    print(i)

    sample_route_list = []
    for i in range(number_of_routes):
        ids = list(range(1, n_cars + 1))
        route = f.generate_routes(ids)
        sample_route_list.append({
            "car_count": len(ids),
            "route": None,
            "route2": route,
            "ids": tuple(ids)
        })
    routes = pd.DataFrame(sample_route_list)

    mix = None
    if mix_type == "same":
        mix = expected_mix

    filtered, pickup, drop = get_filtered_dMatrix_real(origin_dest_table, dMatrix, n_cars)

    data.append({
        "iteration": i,
        "routes": routes,
        "mix": mix,
        "filtered_dmatrix": filtered,
        "pickup_locations": pickup,
        "dropoff_locations": drop
    })

table = pd.DataFrame(data)
table["mix_sorted"] = [tuple(["T0"]+sorted(r["mix"])) for i, r in table.iterrows()]
table.to_pickle(f"files/benchmark_data/{number_of_iteration}_exp_{number_of_routes}_routes_{n_cars}_slots_{prob_type}_{mix_type}.pkl")

all_types = []
for i,r in table.iterrows():
    all_types.extend(r["mix"])

all_type_and_state = f.get_all_type_and_valid_state_type_specific(n_cars, loading_constraints_dict, table.mix_sorted.unique())

#save generated all configurations
fn_all_states = f'files/all_type_and_state/{level}/all_type_and_state_{number_of_iteration}_exp_{number_of_routes}_routes_{n_cars}_slots_{prob_type}_{mix_type}.pickle'
with open(fn_all_states, 'wb') as handle:
    pickle.dump(all_type_and_state, handle, protocol=pickle.HIGHEST_PROTOCOL)