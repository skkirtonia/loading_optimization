import pandas as pd
import numpy as np

# create pickup and drop off location pair
print("generating initial od pairs for benchmark problem")
t1 = pd.read_pickle("files/given/locations_cleaned_fl.pkl")
dist_matrix = pd.read_pickle("files/real_data/distance_matrix_fl_road.pkl")

# Creating a weight based on population (total weight 1)
total_pop = sum(t1.population)
t1["fraction_demand"] = t1.population / total_pop

# create a list of weight for pickup and drop off
pick_w = np.array(list(t1.fraction_demand))
drop_w = np.array(list(t1.fraction_demand))
index = list(t1.index)

origin_dest_pair = []
for i in range(2000):
    # select a pickup location
    select_pick = np.random.choice(index, 1, replace=False, p=pick_w / sum(pick_w))

    # select probabale drop off location ids >100 mile distannce and weight for drop off is not zero
    probable_drop = list(dist_matrix[(dist_matrix[select_pick[0]] > 100)].index)

    # check if there is any probable drop off location is found
    if len(probable_drop) > 0:
        probabale_weight = drop_w[probable_drop]
        # based on the probable drop off location recalculate weights and select a drop off location
        select_drop = np.random.choice(probable_drop, 1, replace=False, p=probabale_weight / sum(probabale_weight))

        origin_dest_pair.append({
            "origin": select_pick[0],
            "destination": select_drop[0],
            "distance": dist_matrix[select_pick[0]][select_drop[0]]
        })
    else:
        print(f"invalid")

origin_dest_table = pd.DataFrame(origin_dest_pair)
origin_dest_table.to_pickle("files/real_data/order_pick_drop_distance_real.pkl")

# Origin from tampa to other parts ------------------------------------------
print("generating initial od pairs for from tampa")
t1 = pd.read_pickle("files/given/locations_cleaned_fl.pkl")
dist_matrix = pd.read_pickle("files/real_data/distance_matrix_fl_road.pkl")
# Creating a weight based on population (total weight 1)
total_pop = sum(t1.population)
t1["fraction_demand"] = t1.population/total_pop
t1_tampa = t1.loc[[2]]
t1_nontampa = t1.drop([2], inplace=False)



# create a list of weight for pickup and drop off
weight = np.array(list(t1.fraction_demand))
index_tampa = list(t1_tampa.index)
index_nontampa = list(t1_nontampa.index)

origin_dest_pair = []

for i in range(2000):
    select_pick = np.random.choice(index_tampa, 1, replace=False,
                                   p=t1_tampa.fraction_demand / sum(t1_tampa.fraction_demand))
    # select probabale drop off location ids >100 mile distannce and weight for drop off is not zero
    probable_drop = list(set(dist_matrix[(dist_matrix[select_pick[0]] > 50)].index) - set(index_tampa))

    probable_drop_table = t1_nontampa.loc[probable_drop]

    # check if there is any probable drop off location is found
    if len(probable_drop) > 0:
        # based on the probable drop off location recalculate weights and select a drop off location
        select_drop = np.random.choice(probable_drop, 1, replace=False,
                                       p=probable_drop_table.fraction_demand / sum(probable_drop_table.fraction_demand))
        # print(select_drop)
        origin_dest_pair.append({
            "origin": select_pick[0],
            "destination": select_drop[0],
            "distance": dist_matrix[select_pick[0]][select_drop[0]]
        })
    else:
        print(f"invalid")
origin_dest_table_from_florida = pd.DataFrame(origin_dest_pair)
origin_dest_table_from_florida.to_pickle("files/real_data/order_pick_from_tampa_drop_distance_real.pkl")


# Destination to Tampa from other parts
print("generating initial od pairs for to tampa")
t1 = pd.read_pickle("files/given/locations_cleaned_fl.pkl")
dist_matrix = pd.read_pickle("files/real_data/distance_matrix_fl_road.pkl")
# Creating a weight based on population (total weight 1)
total_pop = sum(t1.population)
t1["fraction_demand"] = t1.population/total_pop

t1_tampa = t1.loc[[2]]
t1_nontampa = t1.drop([2], inplace=False)

# create a list of weight for pickup and drop off
weight = np.array(list(t1.fraction_demand))
index_tampa = list(t1_tampa.index)
index_nontampa = list(t1_nontampa.index)

origin_dest_pair = []

for i in range(2000):
    # select a pickup location
    select_pick = np.random.choice(index_nontampa, 1, replace=False,
                                   p=t1_nontampa.fraction_demand / sum(t1_nontampa.fraction_demand))
    # select probabale drop off location ids >100 mile distannce and weight for drop off is not zero
    probable_drop = list(set(dist_matrix[(dist_matrix[select_pick[0]] > 50)].index) - set(index_nontampa))

    probable_drop_table = t1.loc[probable_drop]

    # check if there is any probable drop off location is found
    if len(probable_drop) > 0:
        # based on the probable drop off location recalculate weights and select a drop off location
        select_drop = np.random.choice(probable_drop, 1, replace=False,
                                       p=probable_drop_table.fraction_demand / sum(probable_drop_table.fraction_demand))
        # print(select_drop)
        origin_dest_pair.append({
            "origin": select_pick[0],
            "destination": select_drop[0],
            "distance": dist_matrix[select_pick[0]][select_drop[0]]
        })
    else:
        print(f"invalid")
origin_dest_table_to_tampa = pd.DataFrame(origin_dest_pair)
origin_dest_table_to_tampa.to_pickle("files/real_data/order_pick_drop_to_tampa_distance_real.pkl")
