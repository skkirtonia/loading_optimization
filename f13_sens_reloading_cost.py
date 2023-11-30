import matplotlib.pyplot as plt
import pandas as pd
import time
rt_start = time.time()

t_coef = pd.read_pickle(f"files/solution/t_coef_100_exp_250_routes_6_slots_benchmark_same.pkl")
reloading_cost_coefficient = [0,20, 50,100, 200,600]

l = []
for coef in reloading_cost_coefficient:
    for i, r in t_coef.iterrows():
        iteration = r["iteration"]
        mix = r["mix"]
        time0 = r["time0"]
        time1a = r["time1a"]
        time1b = r["time1b"]
        time1c = r["time1c"]
        min_total_cost_0 = r["min_total_cost_0"]
        t1 = r["t1"]

        t1["loading_cost_weighted_p1a"] = t1["cost_p1a"] * coef
        t1["loading_cost_weighted_p1b"] = t1["cost_p1b"] * coef
        t1["loading_cost_weighted_p1c"] = t1["cost_p1c"] * coef

        t1["total_cost_p1a"] = t1["routing_cost"] + t1["loading_cost_weighted_p1a"]
        t1["total_cost_p1b"] = t1["routing_cost"] + t1["loading_cost_weighted_p1b"]
        t1["total_cost_p1c"] = t1["routing_cost"] + t1["loading_cost_weighted_p1c"]

        t1 = t1.sort_values(by="total_cost_p1a")
        t1_opt_1a = t1.iloc[0].to_dict()

        t1 = t1.sort_values(by="total_cost_p1b")
        t1_opt_1b = t1.iloc[0].to_dict()

        t1 = t1.sort_values(by="total_cost_p1c")
        t1_opt_1c = t1.iloc[0].to_dict()

        t1_min_tota_cost_1a = t1_opt_1a["total_cost_p1a"]
        t1_min_tota_cost_1b = t1_opt_1b["total_cost_p1b"]
        t1_min_tota_cost_1c = t1_opt_1a["total_cost_p1c"]

        l.append({
            "iteration": iteration,
            "coef": coef,
            "time0": time0,
            "time1a": time1a,
            "time1b": time1b,
            "time1c": time1c,
            "min_total_cost_0": min_total_cost_0,
            "t1_opt_1a": t1_opt_1a,
            "t1_opt_1b": t1_opt_1b,
            "t1_opt_1c": t1_opt_1c,
            "t1_min_tota_cost_1a": t1_min_tota_cost_1a,
            "t1_min_tota_cost_1b": t1_min_tota_cost_1b,
            "t1_min_tota_cost_1c": t1_min_tota_cost_1c

        })
df = pd.DataFrame(l)
data = df.groupby("coef").agg(
    Policy0 = ("min_total_cost_0", "mean"),
    Policy1A= ("t1_min_tota_cost_1a", "mean"),
    Policy1B= ("t1_min_tota_cost_1b", "mean"),
    Policy1C= ("t1_min_tota_cost_1c", "mean"),
    time0= ("time0", "mean"),
    time1A= ("time1a", "mean"),
    time1B= ("time1b", "mean"),
    time1C= ("time1c", "mean"),
    ).reset_index()

# Compare both policies by total cost and solution time
# plot solution time in x-axis and cost in y-axis for both policy 0 and 1
ax = plt.figure(figsize=(8, 6))

# X_coords = [list(benchmark.time0),list(benchmark.time1_limit), list(benchmark.time1)]
# Y_coords = [list(benchmark.t0_min_tota_cost),list(benchmark.t1_min_tota_cost_limit), list(benchmark.t1_min_tota_cost)]

color = ["black", "black", "red", "black", "black", "black", "black", ]
linestyles = ["dotted", "solid", "dashed", "dashdot", (0, (5, 1)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5))]
for i, r in data.iterrows():
    X_coords = [[r["time0"]], [r["time1B"]], [r["time1A"]]]
    Y_coords = [[r["Policy0"]], [r["Policy1B"]], [r["Policy1A"]]]
    plt.scatter(X_coords[0],
                Y_coords[0],
                s=50,
                c="black",
                marker="s")
    plt.scatter(X_coords[1],
                Y_coords[1],
                s=50,
                c="blue")
    plt.scatter(X_coords[2],
                Y_coords[2],
                s=50,
                c="green")
    plt.plot(X_coords,
             Y_coords,
             color=color[i], linewidth=1, label=f"{int(r['coef'])} mi/reload", linestyle=linestyles[i])
    plt.xlabel("Solution time per experiment (seconds)", size=10)
    plt.ylabel("Total cost (equivalent miles)", size=10)
plt.ylim([1430, 1600])
ax.text(0.15, 0.91, 'Policy 0', horizontalalignment='center', verticalalignment='center')
ax.text(0.33, 0.91, 'Policy 1b', horizontalalignment='center', verticalalignment='center')
ax.text(0.93, 0.91, 'Policy 1a', horizontalalignment='center', verticalalignment='center')

ax.text(0.5, 0.43, 'Benchmark', horizontalalignment='center', verticalalignment='center')
ax.legend(loc='lower left', bbox_to_anchor=(0.1, 0.1))
plt.tight_layout()
plt.savefig("figs/figure13.svg", format="svg")
# plt.savefig("compare policies reloading cost coef 0 1A 1B.png", format="png")
plt.show()

print("runtime = ", (time.time() - rt_start))