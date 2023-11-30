import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pickle
import time
rt_start = time.time()


def generate_node_positions(g, n_stage):
    d = {}
    for i in range(n_stage):
        filtered_nodes = [node for node in g.nodes() if f"-{i}" in node]
        # print(filtered_nodes)
        x = np.arange(len(filtered_nodes))
        x = x - len(x) // 2
        for j, node in enumerate(filtered_nodes):
            pos = ((i + 1) * 100, x[j] * 150)
            d[node] = pos
    return d

# Start ################ generate space state network for Policy 1a ####################
t_coef = pd.read_pickle(f"files/solution/t_coef_100_exp_250_routes_6_slots_benchmark_same.pkl")
selected = t_coef.iloc[30]
t0_opt, t1a_opt, t1b_opt, t1c_opt = selected.t0_opt, selected.t1a_opt, selected.t1b_opt, selected.t1c_opt

data = None
with open('files/graph_info/p1a_graph_3_21_compare_p1a_p0.pickle', 'rb') as f:
    data = pickle.load(f)
shortest_path, shortest_path_length, reload_count, g, route = data
pos = generate_node_positions(g, n_stage=13)

routing_cot_only = "{:.2f}".format(t1a_opt.routing_cost)
total_cost = "{:.2f}".format(t1a_opt.total_cost_p1a)
plt.figure(figsize=(20, 12))
nx.draw_networkx_nodes(g, pos, alpha=.4, node_size=20)
nx.draw_networkx_edges(g, pos, width=.3, arrows=False, alpha=0.1)

pathGraph = nx.path_graph(shortest_path)
path_edges = list(zip(shortest_path, shortest_path[1:]))
edge_labels = {}
for ea in pathGraph.edges():
    edge_labels[ea] = g.edges[ea[0], ea[1]]["length"]
node_labels = {}
for node in shortest_path:
    str_node = node[node.index("(") + 1:node.index(")")].split(",")
    node_labels[node] = ("[" + ",".join(str_node[3:]) + "]" + "\n [" + ",".join(str_node[:3]) + "]").replace("0", "-")

nx.draw(pathGraph, pos, labels=node_labels, with_labels=True, node_size=20)
nx.draw_networkx_edges(pathGraph, pos, edge_color='red', width=3, label=True)
nx.draw_networkx_edge_labels(pathGraph, pos, edge_labels=edge_labels, font_color="r")

route_text = ','.join([f"{vid}+" if action == 1 else f"{vid}-" for action, vid in route])
print(route_text)
plt.annotate(
    f"(b) Policy 1a\nOptimum route: ({route_text})\n# of reloads = {shortest_path_length}\nRouting cost only = {routing_cot_only}\nTotal cost = {total_cost}\n# of vertices = {len(g.nodes())}\n# of edges = {len(g.edges())}",
    xy=(-.01, .96), xycoords="axes fraction", fontsize=12,
    xytext=(30, -50), textcoords="offset points",
    va="center", ha="left")

# Figure 8 in the paper
plt.savefig(f"figs/figure8b.svg", format="svg")
# plt.savefig(f"figs/p1a_graph_3_21_compare_p1a_p0.png", format="png")
plt.show()
# End ################ generate space state network for Policy 1a ####################

# Start ################ generate space state network for Policy 0 ####################
data = None
with open('files/graph_info/p0_graph_new_3_21_compare_p1a_p0.pickle', 'rb') as f:
     data = pickle.load(f)
shortest_path, shortest_path_length, reload_count, g, route_p0 = data
total_cost = "{:.2f}".format(t0_opt.total_cost_p0)
routing_cost = "{:.2f}".format(t0_opt.routing_cost)

pos = generate_node_positions(g, n_stage=13)

node_labels = {}
for node in g.nodes():
    str_node = node[node.index("(") + 1:node.index(")")].split(",")
    node_labels[node] = ("(" + ",".join(str_node[3:]) + "]" + "\n [" + ",".join(str_node[:3]) + ")").replace("0", "-")

plt.figure(figsize=(20, 12))
nx.draw_networkx_nodes(g, pos, alpha=.6, node_size=20)
nx.draw_networkx_edges(g, pos, width=.7, arrows=False, alpha=0.8)
pathGraph = nx.path_graph(shortest_path)
path_edges = list(zip(shortest_path, shortest_path[1:]))
edge_labels = {}

for ea in pathGraph.edges():
    edge_labels[ea] = g.edges[ea[0], ea[1]]["length"]
node_labels_sp = {}

for node in shortest_path:
    str_node = node[node.index("(") + 1:node.index(")")].split(",")
    node_labels_sp[node] = ("[" + ",".join(str_node[3:]) + "]" + "\n [" + ",".join(str_node[:3]) + "]").replace("0",
                                                                                                                "-")

nx.draw(pathGraph, pos, labels=node_labels_sp, with_labels=node_labels_sp, node_size=20)
nx.draw_networkx_edges(pathGraph, pos, edge_color='red', width=3, label=True)
nx.draw_networkx_edge_labels(pathGraph, pos, edge_labels=edge_labels, font_color="r")

route_text = ','.join([f"{vid}+" if action == 1 else f"{vid}-" for action, vid in route_p0])
print(route_text)

plt.annotate(
    f"(a) Policy 0\nOptimum route: ({route_text})\n# of reloads = {shortest_path_length}\nRouting cost only = {routing_cost}\nTotal cost = {total_cost}\n# of nodes = {len(g.nodes())}\n# of edges = {len(g.edges())}",
    xy=(-.01, .96), xycoords="axes fraction", fontsize=12,
    xytext=(30, -50), textcoords="offset points",
    va="center", ha="left")

plt.savefig(f"figs/figure8a.svg", format="svg")
# plt.savefig(f"figs/p0_graph_new_compare_p1a_p0.png", format="png")
plt.show()

print("runtime = ", (time.time() - rt_start))
