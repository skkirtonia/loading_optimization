import matplotlib.pyplot as plt
import geopandas
import pandas as pd
import numpy as np
from adjustText import adjust_text
import time

rt_start = time.time()

states = geopandas.read_file('files/us states data shape/usa-states-census-2014.shp')
order_pick_drop_distance_real = pd.read_pickle("files/realistic/order_pick_drop_distance_real_45_orders.pkl")
southeast = states[states['STUSPS'].isin(['FL', 'GA', 'AL', 'SC', 'NC'])]
location_cleaned = pd.read_pickle("files/given/locations_cleaned.pkl")
route_summary = pd.read_pickle("files/realistic/v1/t_coef_summary_2_sets_800_r_25_a_45_1b_1c_0.pkl")
# the selected routes in the solution
policy1b_sloved = route_summary.loc[np.array([153, 163, 290, 545, 796, 1146, 1229, 1313, 1447, 1462]) - 1]
location_pick_drop = location_cleaned[location_cleaned.population > 0]
location_depot = location_cleaned[location_cleaned.population == 0]

routes_1b = []
point_pick_drop_text_1b = []

for i, r in policy1b_sloved.iterrows():
    automobiles = r["automobiles_covered"]
    route_1b = r["optimum_selection_1b"]["route2"]
    # print(automobiles, route_1b)

    od_route = [0]
    d = {}
    for a, b in route_1b:

        automobile_id = automobiles[b - 1]

        if a > 0:
            origin = int(order_pick_drop_distance_real.loc[automobile_id - 1]["origin"])
            od_route.append(origin)

            data = d.get(origin, [])
            data.append(f"{automobile_id}+")
            d[origin] = data

        else:
            destination = int(order_pick_drop_distance_real.loc[automobile_id - 1]["destination"])
            od_route.append(destination)

            data = d.get(destination, [])
            data.append(f"{automobile_id}-")
            d[destination] = data

    point_pick_drop_text_1b.append(d)
    # print(od_route)
    prev = object()
    od_route = [prev := v for v in od_route if prev != v]
    od_route.append(0)

    # print(od_route)
    routes_1b.append(od_route)

all_order_pick_drop_text = {}
for d in point_pick_drop_text_1b:
    for loc, text in d.items():
        prev_text = all_order_pick_drop_text.get(loc, [])

        prev_text.extend(text)
        all_order_pick_drop_text[loc] = sorted(prev_text)


coordinates = {}
for i, r in location_cleaned.iterrows():
    coordinates[i] = (r["lon"], r["lat"])

fig = plt.figure(1, figsize=(10, 10))
ax = fig.add_subplot()
southeast.apply(lambda x: ax.annotate(text=x.STUSPS, xy=(x.geometry.centroid.coords[0][0], x.geometry.centroid.coords[0][1]* 1.01), ha='center', fontsize=12), axis=1)
southeast.boundary.plot(ax=ax, color='Black', linewidth=.4)
southeast.plot(ax=ax, cmap='Pastel2', figsize=(12, 12))

origins_set = set(order_pick_drop_distance_real.origin)
destination_set = set(order_pick_drop_distance_real.destination)
loc_set = origins_set | destination_set

filtered_pick_drop = location_pick_drop.loc[list(loc_set)]

loc_not_selected = (set(coordinates.keys()) - loc_set) - set([0])

# print(loc_not_selected)
filtered_not_selected = location_pick_drop.loc[list(loc_not_selected)]

plt.scatter(filtered_not_selected.lon, filtered_not_selected.lat, edgecolors='red', facecolors='none',
            label='Areas without orders', s=50)

plt.scatter(filtered_pick_drop.lon, filtered_pick_drop.lat, c='red', label='Areas with orders', s=50)
plt.scatter(location_depot.lon, location_depot.lat, c='black', marker="s", s=50, label='Auto-carrier depot')
color = ["red", "blue", "green", "orange"]

texts = []
for loc_id, text in all_order_pick_drop_text.items():
    x_mul = 1
    y_mul = 1

    full_text = "( " + ",".join(text) + ")"
    texts.append(plt.text(coordinates[loc_id][0] * x_mul, coordinates[loc_id][1] * y_mul,
                          f"({full_text.count('+')}, {full_text.count('-')})", fontsize=14))

adjust_text(texts, only_move={'points': 'y', 'texts': 'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

an3 = ax.annotate(f"# of pickups", xy=(-80.05, 26.35), xycoords="data",
                  xytext=(5, 40), textcoords="offset points",
                  va="center", ha="left",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->"), fontsize=14)
an4 = ax.annotate(f"# of drop-offs", xy=(-79.6, 26.45), xycoords="data",
                  xytext=(50, 10), textcoords="offset points",
                  va="center", ha="left",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->"), fontsize=14)
ax.legend(loc='center right', bbox_to_anchor=(0.5, 0.12), fontsize='14')

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()

plt.savefig("figs/figure16.svg", format="svg")
plt.show()
print("runtime = ", (time.time()-rt_start))
