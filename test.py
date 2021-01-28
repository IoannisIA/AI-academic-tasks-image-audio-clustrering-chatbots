from itertools import product
import networkx as nx
import matplotlib.pyplot as plt

delay_table = [0, 0, 0, 0, 0]
delay = [0, 1]
delay_value = 30
cases = list(product(delay, delay, delay, delay, delay))


def make_a_child(old_delay_table):
    new_delay_table = []
    G.add_node(tuple(old_delay_table), state=sum(old_delay_table))
    min_cost = 0
    delay_value = 30
    current_state = G.nodes[tuple(old_delay_table)]['state']
    for case in cases:
        temp_delay_table = old_delay_table.copy()
        l = list(case)
        temp_new_state = current_state
        for index in range(5):
            if l[index] == 1:
                temp_delay_table[index] += delay_value
                temp_new_state = + delay_value
        if temp_new_state - current_state < min_cost:
            min_cost = temp_new_state - current_state
            new_delay_table = temp_delay_table


    G.add_node(tuple(new_delay_table), state=sum(new_delay_table))

    G.add_edge(tuple(old_delay_table), tuple(new_delay_table), weight=0.6)

    return new_delay_table






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

new_delay_table = make_a_child(delay_table)

make_a_child(new_delay_table)

plot_graph()
