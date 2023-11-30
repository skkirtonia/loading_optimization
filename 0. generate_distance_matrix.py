import pandas as pd
import numpy as np
import googlemaps

t1 = pd.read_pickle("files/given/locations_cleaned_fl.pkl")

print("generating roadway distance matrix ")
api_key = "{APIKEY}"
url = "https://maps.googleapis.com/maps/api/distancematrix/json?origins=40.6655101%2C-73.89188969999998&destinations=40.659569%2C-73.933783%7C40.729029%2C-73.851524%7C40.6860072%2C-73.6334271%7C40.598566%2C-73.7527626&key=" + api_key

index_list = [(0, 10), (10, 20), (20, 22)]
gmaps = googlemaps.Client(key=api_key)

# For each pair of batches, query for the distance matrix and save save them
l = {}
for a1, a2 in index_list:
    for b1, b2 in index_list:
        o = t1[a1:a2]
        d = t1[b1:b2]
        origins = [(r["lat"], r["lon"]) for i, r in o.iterrows()]
        destinations = [(r["lat"], r["lon"]) for i, r in d.iterrows()]
        result = gmaps.distance_matrix(origins, destinations, mode='driving', units="imperial")
        l[(a1, a2, b1, b2)] = result

# For each of the responses received I find the distance information and save in a dataframe
data = []
for a1, a2 in index_list:
    for b1, b2 in index_list:
        result = l[(a1, a2, b1, b2)]
        for i, row in enumerate(result["rows"]):
            for j, col in enumerate(row["elements"]):
                data.append({"i": i + a1, "j": j + b1, "dist": col["distance"]["value"]})

df = pd.DataFrame(data)
# from the dataframe I build a distance matrix
dm = df.pivot_table(index="i", columns="j", values="dist")
# converting the unit ot miles
dm_mile = dm * 0.000621371
dm_mile.to_pickle("files/real_data/distance_matrix_fl_road.pkl")