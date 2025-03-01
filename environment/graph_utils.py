import numpy as np
import networkx as nx
from networkx.algorithms import bipartite

import torch
import torch_geometric
from torch_geometric.data import Data

from environment.bipartite import BipartiteData

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def num_to_chr(num):
    try:  # see if list
        iterator = iter(num)
    except TypeError: # not iterable
        return chr(ord('a') + num)
    else: # iterable
        return [chr(ord('a') + j) for j in num]
    

def visualize_graph(rmab, out_dir, good_actions=None, bad_actions=None):
    """ given a torch_geometric graph, visualize it 
    assuming bipartite structure, with node positions given by node_pos 
    good_actions and bad_actions - list of actions to highlight"""

    labels = {}

    # create bipartite graph
    B = nx.Graph()
    B.add_nodes_from(range(rmab.action_dim), bipartite=0)  # add actions
    B.add_nodes_from(num_to_chr(range(rmab.n_arms)), bipartite=1)  # add nodes
    for i in range(rmab.action_dim):
        for j in range(rmab.n_arms):
            if rmab.weights_p[j][i] != 0:
                print(i,j)
                B.add_edge(i, num_to_chr(j), weight_p=rmab.weights_p[j][i], weight_q=0)
                labels[(i, num_to_chr(j))] = f'{rmab.weights_p[j][i]:.2f}'


    # G = torch_geometric.utils.to_networkx(data, to_undirected=True)
    # # nx.draw(g)
    node_color = ['paleturquoise'] * rmab.action_dim + ['goldenrod'] * rmab.n_arms # node fill color

    if good_actions is not None:
        for i in good_actions:
            node_color[i] = 'green'
    if bad_actions is not None:
        for i in bad_actions:
            node_color[i] = 'red'

    edgecolors = 'r' # node outline color
    linewidths = 0

    theta_node_labels = {num_to_chr(j): f'{rmab.theta_p[j]:.2f}' for j in range(rmab.n_arms)}
    node_pos_higher = {k: (rmab.node_pos[k][0], rmab.node_pos[k][1] + 0.12) for k in rmab.node_pos.keys()}

    plt.figure()
    nx.draw_networkx(B, rmab.node_pos, with_labels=True, node_color=node_color, edgecolors=edgecolors, linewidths=linewidths)
    edge_labels = nx.draw_networkx_edge_labels(B, rmab.node_pos, edge_labels=labels, alpha=0.5, label_pos=0.7)
    nx.draw_networkx_labels(B, node_pos_higher, labels=theta_node_labels, font_size=8, alpha=0.5, verticalalignment='bottom', horizontalalignment='right')
    plt.title('bipartite graph with edge weights $(\\theta_p, \\theta_q)$')
    plt.savefig(f'{out_dir}/plot_bipartite_graph_{rmab.link_type}_n{rmab.n_arms}_j{rmab.action_dim}.png')
    # plt.show()
    plt.close()


def make_geometric(weights_p, init_state):
    """ create torch_geometric graph """
    n_arms, action_dim = weights_p.shape

    # node features
    x_action = torch.Tensor(np.arange(action_dim)).unsqueeze(1)
    x_arm    = torch.Tensor(np.arange(n_arms)).unsqueeze(1)
    
    # set up edges
    edge_index = []
    edge_attr  = []

    for i in range(action_dim):
        for j in range(n_arms):
            if weights_p[j][i] != 0:
                edge_index.append((i, j))
                edge_attr.append((weights_p[j][i]))

    print('edge index', edge_index)

    edge_index = torch.tensor(edge_index, dtype=torch.int64, device=device).T
    edge_attr  = torch.tensor(edge_attr, dtype=torch.float, device=device).T


    data = BipartiteData(x_action=x_action, x_arm=x_arm, edge_index=edge_index, edge_attr=edge_attr)

    # ensure valid
    data.validate(raise_on_error=True)

    return data


if __name__ == '__main__':
    # create very simple graph with 3 nodes
    edge_index = torch.tensor([[0, 1, 1, 2],
                            [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    
    data = torch_geometric.data.Data(x=x, edge_index=edge_index)

    # draw graph
    visualize_graph(data)