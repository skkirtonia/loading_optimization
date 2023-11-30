from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

rt_start = time.time()
files = {
    "Benchmark": "t_coef_100_exp_250_routes_6_slots_benchmark_same",
    "To Tampa only": "t_coef_100_exp_250_routes_6_slots_to_tampa_same",
    "From Tampa only": "t_coef_100_exp_250_routes_6_slots_from_tampa_same",
}
legend_elements_small_0_1a_1b = [
    Line2D([0], [0], marker='s', color='w', label='Policy 0', markerfacecolor='black', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Policy 1b', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Policy 1a', markerfacecolor='green', markersize=10)
]
def print_scatter2(compare, X_coords, Y_coords, loc, anchor, anchor_pol=(0.5, 0.0, 0.45, 0.95)):
    fig, ax = plt.subplots(1, figsize=(7, 6))

    color = {
        "Benchmark": "red",
        "To Tampa only": "blue",
        "From Tampa only": "green",
        "FL": "orange",
        "NC to FL": "magenta",
        "Pick First": "black"}
    for i, x in enumerate(compare):
        plt.scatter(X_coords[i][0],
                    Y_coords[i][0],
                    s=50,
                    c="black", marker="s")
        plt.scatter(X_coords[i][1],
                    Y_coords[i][1],
                    s=50,
                    c="blue")
        plt.scatter(X_coords[i][2],
                    Y_coords[i][2],
                    s=50,
                    c="green")

        ax.plot(X_coords[i],
                Y_coords[i],
                color=color[x], linewidth=2, label=compare[i])
        plt.xlabel("Solution time per experiment(seconds)", size=10)
        plt.ylabel("Total cost (equivalent miles)", size=10)

    return ax


def compare_plot2(compare, loc, anchor, anchor_pol=(0.5, 0.0, 0.45, 0.95)):
    X_coords = []
    Y_coords = []
    for x in compare:
        t_coef = pd.read_pickle(f"files/solution/{files[x]}.pkl")
        X_coords.append([[t_coef.time0.mean()], [t_coef.time1b.mean()], [t_coef.time1a.mean()]])
        Y_coords.append(
            [[t_coef.min_total_cost_0.mean()], [t_coef.min_total_cost_1b.mean()], [t_coef.min_total_cost_1a.mean()]])
    # print(X_coords)
    # print(Y_coords)
    ax = print_scatter2(compare, X_coords, Y_coords, loc, anchor, anchor_pol)
    return ax


data_mean = []

for k, v in files.items():
    t_coef = pd.read_pickle(f"files/solution/{v}.pkl")
    d = dict(t_coef[["min_total_cost_0", "min_total_cost_1b", "min_total_cost_1a", 'time0', 'time1a', 'time1b']].mean())

    d["case"] = k
    data_mean.append(d)

mean_table = pd.DataFrame(data_mean)


mean_table["0_1b_cost_saving"] = (mean_table["min_total_cost_0"] - mean_table["min_total_cost_1b"]) / mean_table[
    "min_total_cost_0"]
cost_saving = dict(zip(mean_table.case, mean_table["0_1b_cost_saving"] * 100))

compare = ["Benchmark", 'To Tampa only', 'From Tampa only']

ax = compare_plot2(compare, loc="lower right", anchor=(0.97, 0.23))
# print(ax.get_legend_handles_labels())
an2 = ax.annotate(f"{round(cost_saving['Benchmark'], 1)}% cost reduction", xy=(15, 1535), xycoords="data",
                  xytext=(30, 30), textcoords="offset points",
                  va="center", ha="left",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->"))
an3 = ax.annotate(f"{round(cost_saving['To Tampa only'], 1)}% cost reduction", xy=(15, 1225), xycoords="data",
                  xytext=(30, 20), textcoords="offset points",
                  va="center", ha="left",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->"))

an4 = ax.annotate(f"{round(cost_saving['From Tampa only'], 1)}% cost reduction", xy=(13, 1195), xycoords="data",
                  xytext=(42, -38), textcoords="offset points",
                  va="center", ha="right",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->"))

ax.legend(loc=(.7, .45))
ax2 = ax.twinx()
ax2.legend(handles=legend_elements_small_0_1a_1b, loc='upper right', bbox_to_anchor=(0.5, 0.0, 0.40, 0.73))
ax2.get_yaxis().set_ticks([])
plt.tight_layout()
plt.savefig(f"figs/figure15.svg", format="svg")

print("runtime = ", (time.time() - rt_start))
plt.show()
