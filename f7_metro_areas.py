import matplotlib.pyplot as plt
import geopandas
import pandas as pd
import time

rt_start = time.time()
states = geopandas.read_file('files/given/us states data shape/usa-states-census-2014.shp')
southeast = states[states['STUSPS'].isin(['FL'])]

metros = geopandas.read_file('files/given/cb_2018_us_cbsa_500k/cb_2018_us_cbsa_500k.shp')
metros_fl = metros[metros['NAME'].str.contains("FL")]

selected_metro_fl = [
    "Miami-Fort Lauderdale-West Palm Beach, FL",
    "Tampa-St. Petersburg-Clearwater, FL",
    "Orlando-Kissimmee-Sanford, FL",
    "Jacksonville, FL",
    "North Port-Sarasota-Bradenton, FL",
    "Cape Coral-Fort Myers, FL",
    "Lakeland-Winter Haven, FL",
    "Deltona-Daytona Beach-Ormond Beach, FL",
    "Palm Bay-Melbourne-Titusville, FL",
    "Pensacola-Ferry Pass-Brent, FL",
    "Port St. Lucie, FL",
    "Tallahassee, FL",
    "Naples-Immokalee-Marco Island, FL",
    "Ocala, FL",
    "Gainesville, FL",
    "Crestview-Fort Walton Beach-Destin, FL",
    "Panama City, FL",
    "Punta Gorda, FL",
    "Sebastian-Vero Beach, FL",
    "Homosassa Springs, FL",
    "Sebring, FL"
]

metros_shape = metros_fl[metros_fl.NAME.isin(selected_metro_fl)]
metros_shape = metros_shape.sample(n=len(metros_shape))
location_cleaned = pd.read_pickle("files/given/locations_cleaned_fl.pkl")
location_pick_drop = location_cleaned[location_cleaned.population>0]
location_depot = location_cleaned[location_cleaned.population==0]

fig = plt.figure(1, figsize=(12,8))
ax = fig.add_subplot()
southeast.boundary.plot(ax=ax, color='Black', linewidth=.4)
southeast.plot(ax=ax, color=['whitesmoke'])

metros_shape.boundary.plot(ax=ax, color='black', linewidth=.4)
metros_shape.plot(ax=ax, cmap='Set3')

plt.scatter(location_pick_drop.lon, location_pick_drop.lat, c='red', label = 'Population centroid',s = 50)
plt.scatter(location_depot.lon, location_depot.lat, c='blue', marker="s", s =30, label= 'Auto-carrier depot')

an3 = ax.annotate(f"Jacksonville Metro", xy=( -81.656004, 30.508815), xycoords="data",
                  xytext=(5, 40), textcoords="offset points",
                  va="center", ha="left",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->"))


an3 = ax.annotate(f"Orlando-Kissimmee-Sanford Metro", xy=(-81.367898, 28.748823), xycoords="data",
                  xytext=(-300, 20), textcoords="offset points",
                  va="center", ha="left",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->"))

an3 = ax.annotate(f"Tampa-St. Petersburg-Clearwater Metro", xy=( -82.446247, 28.288094), xycoords="data",
                  xytext=(-300, 0), textcoords="offset points",
                  va="center", ha="left",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->"))

an3 = ax.annotate(f"Miami-Fort Lauderdale-West Palm Beach Metro", xy=(-80.561178, 26.085506), xycoords="data",
                  xytext=(-350, -30), textcoords="offset points",
                  va="center", ha="left",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->"))

ax.legend(loc='center right', bbox_to_anchor=(0.5,0.12),fontsize='13')

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()

# Figure 7 in the paper
plt.savefig("figs/figure7.svg", format="svg")
print("runtime = ", (time.time()-rt_start))
plt.show()