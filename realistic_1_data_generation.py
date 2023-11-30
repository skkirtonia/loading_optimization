
import pandas as pd
import numpy as np
import random
import Functions as f
import time

rt_start = time.time()

# for different experiments change the values of number_of_automobile and number_of_routes
number_of_automobile = 45
number_of_routes = 10


number_of_auto_carrier = 2
number_of_sets = 100
types = ["T1", "T2", "T3"]
types = ["T1"] * 6 + ["T2"] * 8 + ["T3"] * 10
automobile_set = list(range(1, number_of_automobile + 1))
autocarrier_set = list(range(1, number_of_auto_carrier + 1))
automobile_types = {auto_id: random.choice(types) for auto_id in automobile_set}

auto_carrier_lavel = {1: "single_level", 2: "bilevel"}
auto_carrier_n_slot = {1: 5, 2: 6}
car_count_in_autocarrier = {1: [3, 4, 5], 2: [4, 5, 6]}
auto_carrier_first_level_slot = {1: 5, 2: 3}
auto_carrier_routing_cost = {1: 2.0, 2: 3.0}  # dollar per mile
auto_carrier_cost_per_reload = {1: 50 / 2.0, 2: 100 / 3.0}  # mile per reload reload
version = "v1"


origin_dest_table = pd.read_pickle(f"files/realistic/order_pick_drop_distance_real_{number_of_automobile}_orders.pkl")
dMatrix = pd.read_pickle('files/real_data/distance_matrix_road.pkl')


origin_dest_table["index"] = list(range(1, number_of_automobile + 1))
origin_dest_table = origin_dest_table.set_index("index", drop=True)


def get_filtered_dMatrix_real(n_cars, dMatrix, automobiles):
    depot = [0]
    pickup = [int(origin_dest_table.loc[a]["origin"]) for a in automobiles]
    drop = [int(origin_dest_table.loc[a]["destination"]) for a in automobiles]

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

l = []
for auto_carrier in autocarrier_set:
    print(auto_carrier)
    for s in range(number_of_sets):
        n_slots = auto_carrier_n_slot[auto_carrier]
        n_cars = random.choice(car_count_in_autocarrier[auto_carrier])
        automobiles = random.sample(automobile_set, n_cars)
        mix = ["T0"] + [automobile_types[auto_id] for auto_id in automobiles]
        sorted_mix = tuple(sorted(mix))

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
        sample_route = pd.DataFrame(sample_route_list)

        filtered, pickup, drop = get_filtered_dMatrix_real(n_cars, dMatrix, automobiles)

        constraints = pd.read_csv(
            f"files/given/constraints/{auto_carrier_lavel[auto_carrier]}/loading_constraint_{n_slots}_car.csv",
            index_col="index")
        constraints = constraints.fillna(0)
        loading_constraints = constraints.astype(int)
        loading_constraints_dict = constraints.to_dict()

        l.append({
            "auto_carrier": auto_carrier,
            "auto_carrier_type": auto_carrier_lavel[auto_carrier],
            "routing_cost": auto_carrier_routing_cost[auto_carrier],
            "reloading_cost": auto_carrier_cost_per_reload[auto_carrier] * auto_carrier_routing_cost[auto_carrier],
            "auto_set": s,
            "n_slots": auto_carrier_n_slot[auto_carrier],
            "n_slots_level1": auto_carrier_first_level_slot[auto_carrier],
            "n_cars": n_cars,
            "automobiles": automobiles,
            "routes": sample_route,
            "mix": mix,
            "filtered_dmatrix": filtered,
            "pickup_locations": pickup,
            "dropoff_locations": drop,
            "mix_sorted": sorted_mix,
            "loading_constraints_dict": loading_constraints_dict

        })

table = pd.DataFrame(l)

table.to_pickle(f"files/realistic/{version}/routes_ac_{number_of_auto_carrier}_sets_{number_of_sets}_r_{number_of_routes}_a_{number_of_automobile}.pkl")
print("runtime = ", (time.time()-rt_start))