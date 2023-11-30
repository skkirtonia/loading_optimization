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
        #print(filtered_nodes)
        x = np.arange(len(filtered_nodes))
        x = x-len(x)//2
        for j, node in enumerate(filtered_nodes):
            pos=((i+1)*100,x[j]*150)
            d[node]=pos
    return d
def generate_node_positions_and_labels(stage_nodes, n_stage):
    d = {}
    l = {}
    for i in range(n_stage):
        filtered_nodes = stage_nodes[i]
        #print(filtered_nodes)
        x = np.arange(len(filtered_nodes))
        x = x-len(x)//2
        pos = [((i+1)*100,each_x*150) for each_x in x]
        pos = sorted(pos, key=lambda item: item[1], reverse=True)
        for j, (node, label) in enumerate(filtered_nodes.items()):
            d[str(node)+"-"+str(i)]=pos[j]
            l[str(node)+"-"+str(i)]=str(list(node)).replace('0','-')+f"  {label}"
    return d, l

# Start------- generate sample ssn for Policy 1a
route = [(1, 2), (1, 3), (1, 4), (1, 1), (-1, 3), (-1, 2), (-1, 1), (-1, 4)]
date = "3_25"
data = None
with open(f'files/graph_info/p1a_graph_{date}_slot4.pickle', 'rb') as f:
     data = pickle.load(f)
print(f'files/graph_info/p1a_graph_{date}_slot4.pickle')
shortest_path, shortest_path_length, reload_count, A = data
pos = generate_node_positions(A, n_stage=9)
g = A
plt.figure(figsize=(20, 12))
node_labels = {}
for node in g:
    str_node = node[node.index("(") + 1:node.index(")")].split(",")
    node_labels[node] = ("[" + ','.join(str_node) + "]").replace("0", "-")

nx.draw_networkx_nodes(g, pos, alpha=.4, node_size=20)
nx.draw_networkx_edges(g, pos, width=.3, arrows=False, alpha=0.8)
nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=14)

shortest_path = ['(0, 0, 0, 0)-0', '(0, 0, 0, 2)-1', '(0, 0, 3, 2)-2', '(0, 4, 3, 2)-3', '(1, 4, 3, 2)-4',
                 '(2, 1, 4, 0)-5', '(0, 1, 4, 0)-6', '(0, 0, 4, 0)-7', '(0, 0, 0, 0)-8']
pathGraph = nx.path_graph(shortest_path)
path_edges = list(zip(shortest_path, shortest_path[1:]))
edge_labels = {}
for ea in pathGraph.edges():
    edge_labels[ea] = g.edges[ea[0], ea[1]]["length"]

# nx.draw(pathGraph,pos, labels=node_labels, with_labels = True, node_size = 20)
nx.draw_networkx_edges(pathGraph, pos, edge_color='red', width=3, label=True)
nx.draw_networkx_edge_labels(pathGraph, pos, edge_labels=edge_labels, font_color="r", font_size=14)

route_text = ','.join([f"{vid}+" if action == 1 else f"{vid}-" for action, vid in route])
print(route_text)
plt.annotate(
    f"(a) Policy 1a\nRoute: ({route_text})\n# of reloads = {shortest_path_length}\n# of vertices = {len(g.nodes())}\n# of edges = {len(g.edges())}",
    xy=(-.01, .96), xycoords="axes fraction", fontsize=14,
    xytext=(30, -50), textcoords="offset points",
    va="center", ha="left")

plt.savefig(f"figs/figure12a.svg", format="svg")
plt.show()

# End------- generate sample ssn for Policy 1a

# Start------- generate sample ssn for Policy 1b
data = None
with open(f'files/graph_info/p1b_graph_3_28_slot4.pickle', 'rb') as f:
     data = pickle.load(f)
shortest_path, shortest_path_length, reload_count, g = data

plt.figure(figsize=(20, 12))
node_labels = {}
for node in g:
    str_node = node[node.index("(") + 1:node.index(")")].split(",")
    node_labels[node] = ("[" + ','.join(str_node) + "]").replace("0", "-")

remove = [node for node, degree in g.degree() if degree < 1]
print(remove)
g.remove_nodes_from(remove)

nx.draw_networkx_nodes(g, pos, alpha=.4, node_size=20)
nx.draw_networkx_edges(g, pos, width=.5, arrows=False, alpha=0.8)
nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=14)

# print(edge_labels)
# nx.draw_networkx_edge_labels(A, pos, edge_labels = edge_labels)

shortest_path = ['(0, 0, 0, 0)-0', '(0, 0, 0, 2)-1', '(0, 0, 3, 2)-2', '(0, 4, 3, 2)-3', '(1, 4, 3, 2)-4',
                 '(1, 4, 0, 2)-5', '(1, 4, 0, 0)-6', '(0, 4, 0, 0)-7', '(0, 0, 0, 0)-8']
pathGraph = nx.path_graph(shortest_path)
path_edges = list(zip(shortest_path, shortest_path[1:]))
edge_labels = {}
for ea in pathGraph.edges():
    edge_labels[ea] = g.edges[ea[0], ea[1]]["length"]

# nx.draw(pathGraph,pos, labels=node_labels, with_labels = True, node_size = 20)
nx.draw_networkx_edges(pathGraph, pos, edge_color='red', width=3, label=True)
nx.draw_networkx_edge_labels(pathGraph, pos, edge_labels=edge_labels, font_color="r", font_size=14)

route_text = ','.join([f"{vid}+" if action == 1 else f"{vid}-" for action, vid in route])
print(route_text)
plt.annotate(
    f"(b) Policy 1b\nRoute: ({route_text})\n# of reloads = {shortest_path_length}\n# of vertices = {len(g.nodes())}\n# of edges = {len(g.edges())}",
    xy=(-.01, .96), xycoords="axes fraction", fontsize=14,
    xytext=(30, -50), textcoords="offset points",
    va="center", ha="left")

plt.savefig(f"figs/figure12b.svg", format="svg")
plt.show()
# End------- generate sample ssn for Policy 1b

print("runtime = ", (time.time() - rt_start))