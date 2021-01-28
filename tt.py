import networkx as nx
import matplotlib.pyplot as plt

"""
G=nx.path_graph(5)
print(nx.astar_path(G,0,4))
G=nx.grid_graph(dim=[3,3])  # nodes are two-tuples (x,y)

def dist(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

print(nx.astar_path(G,(0,0),(2,2),dist))
nx.draw(G)
plt.show()
"""

def make_a_child(a, b, c, d, e):
    G.add_node((a, b, c, d, e), a=a, b=b, c=c, d=d, e=e, state=a + b + c + d + e)
    min_cost = 0
    delay = 30
    current_state = G.nodes[(a, b, c, d, e)]['state']
    for case in range(6):
        if case == 0:
            new_state = current_state
            #G.add_edge((a, b, c, d, e), (a, b, c, d, e), weight=0.1)
            min_cost = new_state - current_state
            next_a = a
            next_b = b
            next_c = c
            next_d = d
            next_e = e
        elif case == 1:
            new_a = a + delay
            new_state = current_state + delay
            #G.add_edge((a, b, c, d, e), (new_a, b, c, d, e), weight=0.1)
            if new_state - current_state < min_cost:
                min_cost = new_state - current_state
                next_a = new_a
                next_b = b
                next_c = c
                next_d = d
                next_e = e







    G.add_node((next_a, next_b, next_c, next_d, next_e), a=next_a, b=next_b, c=next_c, d=next_d, e=next_e,
               state=next_a + next_b + next_c + next_d + next_e)

    G.add_edge((a, b, c, d, e), (next_a, next_b, next_c, next_d, next_e), weight=0.6)

    return next_a, next_b, next_c, next_d, next_e












def plot_graph():

    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] < 0.5]
    pos = nx.spring_layout(G)  # positions for all nodes


    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=500)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=3)
    nx.draw_networkx_edges(G, pos, edgelist=esmall, width=1, alpha=0.5, edge_color='red', style='dashed')

    # labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    plt.axis('off')
    plt.show()









G = nx.Graph()
new_a, new_b, new_c, new_d, new_e = make_a_child(0, 0, 0, 0, 0)

for i in range(7):
    new_a, new_b, new_c, new_d, new_e = make_a_child(new_a, new_b, new_c, new_d, new_e)


plot_graph()