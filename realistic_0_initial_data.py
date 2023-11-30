import pandas as pd
import numpy as np

number_of_orders = 45
t1 = pd.read_csv("files/real_data/locations_cleaned.csv")
dist_matrix = pd.read_pickle("files/real_data/distance_matrix_road.pkl")
# Creating a weight based on population (total weight 1)
total_pop = sum(t1.population)
t1["fraction_demand"] = t1.population/total_pop

# create a list of weight for pickup and drop off
pick_w = np.array(list(t1.fraction_demand))
drop_w = np.array(list(t1.fraction_demand))
index = list(t1.index)

origin_dest_pair = []

for i in range(number_of_orders):
    # select a pickup location
    select_pick = np.random.choice(index, 1, replace = False, p = pick_w/sum(pick_w))

    #select probabale drop off location ids >100 mile distannce and weight for drop off is not zero
    probable_drop = list(dist_matrix[(dist_matrix[select_pick[0]]>100)].index)

    # check if there is any probable drop off location is found
    if len(probable_drop)>0:
        probabale_weight = drop_w[probable_drop]
        # based on the probable drop off location recalculate weights and select a drop off location
        select_drop = np.random.choice(probable_drop, 1, replace = False, p = probabale_weight/sum(probabale_weight))

        origin_dest_pair.append({
            "origin": select_pick[0],
            "destination":select_drop[0],
            "distance":dist_matrix[select_pick[0]][select_drop[0]]
        })
    else:
        print(f"invalid")

origin_dest_table = pd.DataFrame(origin_dest_pair)
origin_dest_table.to_pickle(f"files/realistic/order_pick_drop_distance_real_{number_of_orders}_orders.pkl")