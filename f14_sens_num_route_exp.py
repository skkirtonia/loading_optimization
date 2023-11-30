from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
rt_start = time.time()

data = pd.read_csv("files/organized/sensitivity n_route per experiment3.csv")
legend_elements = [Line2D([0], [0], marker='s', color='black', label='Policy 0',
                          markerfacecolor='black', markersize=9, linestyle = 'None'),
                   Line2D([0], [0], marker='o', color='blue', label='Policy 1b',
                          markerfacecolor='blue', markersize=9, linestyle = 'None'),
                   Line2D([0], [0], marker='o', color='green', label='Policy 1a',
                          markerfacecolor='green', markersize=9, linestyle = 'None')]


fig, ax = plt.subplots(figsize=(7,5))
for i, r in data.iterrows():
    n_route = r["n_route"]
    min_cost_p1a = r["min_cost_p1a"]
    min_cost_p1b = r["min_cost_p1b"]
    min_cost_p1c = r["min_cost_p0"]
    sol_time_p1a = r["sol_time_p1a"]
    sol_time_p1b = r["sol_time_p1b"]
    sol_time_p1c = r["sol_time_p0"]
    x_data = [sol_time_p1c, sol_time_p1b, sol_time_p1a]
    y_data = [min_cost_p1c, min_cost_p1b, min_cost_p1a]
    sns.lineplot(x= x_data, y = y_data, label = f"{round(n_route)} routes per experiment", ax = ax)
    color = ["black", "blue", "green"]
    markers = ["s", "o", "o"]
    for j, (x, y) in enumerate(zip(x_data, y_data)):
        ax.plot(x,y, marker = markers[j], color = color[j], markersize = 7)
l1 = plt.legend()
ax.legend(handles = legend_elements, loc= "center right")
plt.gca().add_artist(l1)

ax.set_xlabel("Solution time (seconds)")
ax.set_ylabel("Total cost (equivalent miles)")
ax.text(75, 1500, 'Benchmark', horizontalalignment='center', verticalalignment='center')
plt.tight_layout()
plt.savefig("figs/figure14.svg", format="svg")
# plt.savefig("compare time-cost changing routes per experiment.pdf", format="pdf")
plt.show()

print("runtime = ", (time.time() - rt_start))