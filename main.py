import matplotlib
matplotlib.use('TkAgg')  

import networkx as nx
import matplotlib.pyplot as plt


G = nx.Graph()


G.add_nodes_from([1, 2, 3])


G.add_edge(1, 2)
e = (2, 3)
G.add_edge(*e)


plt.figure(figsize=(5, 5))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')


plt.show()
