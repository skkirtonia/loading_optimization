import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
rt_start = time.time()

def print_scatter_0_1a_1b(X_coords, Y_coords):
    ax = plt.figure(figsize=(8, 6))

    plt.scatter(X_coords[0],
                Y_coords[0],
                s=50,
                marker='s',
                c="black", label="Policy 0")

    plt.scatter(X_coords[1],
                Y_coords[1],
                s=50,
                c="blue", label="Policy 1b")

    plt.scatter(X_coords[2],
                Y_coords[2],
                s=50,
                c="green", label="Policy 1a")
    plt.plot(X_coords,
             Y_coords,
             color='gray', linewidth=.5)
    plt.xlabel("Solution time per experiment (seconds)", size=10)

    plt.ylim([700, 2800])

    plt.ylabel("Total cost (equivalent miles)", size=10)
    #     ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.5, 0.0, 0.45, 0.95))
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/figure10.svg", format="svg")
    # plt.savefig("compare policies 0 1A 1B.png", format="png")
    plt.show()


def print_scatter_0_1a_1c(X_coords, Y_coords):
    ax = plt.figure(figsize=(8, 6))

    plt.scatter(X_coords[0],
                Y_coords[0],
                s=50,
                marker='s',
                c="black", label="Policy 0")
    plt.scatter(X_coords[1],
                Y_coords[1],
                s=50,
                c="orange", label="Policy 1c")

    plt.scatter(X_coords[2],
                Y_coords[2],
                s=50,
                c="green", label="Policy 1a")
    plt.plot(X_coords,
             Y_coords,
             color='gray', linewidth=.5)
    plt.xlabel("Solution time per experiment (seconds)", size=10)
    plt.ylabel("Total cost (equivalent miles)", size=10)
    plt.ylim([700, 2800])
    #     ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.5, 0.0, 0.45, 0.95))
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/figure11.svg", format="svg")
    # plt.savefig("compare policies 0 1A 1C.png", format="png")
    plt.show()


def print_scatter_p0_p1a(X_coords, Y_coords):
    ax = plt.figure(figsize=(8, 6))

    plt.scatter(X_coords[0],
                Y_coords[0],
                s=50,
                marker='s',
                c="black", label="Policy 0")

    plt.scatter(X_coords[1],
                Y_coords[1],
                s=50,
                c="green", label="Policy 1a")
    plt.plot(X_coords,
             Y_coords,
             color='gray', linewidth=.5)

    plt.ylim([700, 2800])
    plt.xlabel("Solution time per experiment (seconds)", size=10)
    plt.ylabel("Total cost (equivalent miles)", size=10)
    #     ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.5, 0.0, 0.45, 0.95))
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/figure9.svg", format="svg")
    # plt.savefig("compare policies p0_p1a.png", format="png")
    plt.show()

t_coef = pd.read_pickle("files/solution/t_coef_100_exp_250_routes_6_slots_benchmark_same.pkl")
X_coords = [
    list(t_coef.time0), #black
    list(t_coef.time1a) # green
]
Y_coords = [
    list(t_coef.min_total_cost_0),
    list(t_coef.min_total_cost_1a)
]
print_scatter_p0_p1a(X_coords, Y_coords)

X_coords = [
    list(t_coef.time0), #black
    list(t_coef.time1b), # blue
    list(t_coef.time1a) # green
]
Y_coords = [
    list(t_coef.min_total_cost_0),
    list(t_coef.min_total_cost_1b),
    list(t_coef.min_total_cost_1a)
]
print_scatter_0_1a_1b(X_coords, Y_coords)

X_coords = [
    list(t_coef.time0), #black
    list(t_coef.time1c), # orange
    list(t_coef.time1a) # green
]
Y_coords = [
    list(t_coef.min_total_cost_0),
    list(t_coef.min_total_cost_1c),
    list(t_coef.min_total_cost_1a)
]
print_scatter_0_1a_1c(X_coords, Y_coords)

print("runtime = ", (time.time() - rt_start))