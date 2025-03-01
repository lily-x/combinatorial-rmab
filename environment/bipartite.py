import numpy as np
import torch
from torch_geometric.data import Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# define bipartite graph
class BipartiteData(Data):
    def __init__(self, x_action=None, x_arm=None, edge_index=None, edge_attr=None):
        super().__init__()
        self.x_action   = x_action
        self.x_arm      = x_arm
        self.edge_index = edge_index
        self.edge_attr  = edge_attr

        self.num_nodes = self.num_nodes

    def get_arm(self):
        return self.x_arm
    
    def get_action(self):
        return self.x_action

    @property
    def num_actions(self) -> int:
        return self.x_action.size(0)
    
    @property
    def num_arms(self) -> int:
        return  self.x_arm.size(0)

    @property
    def num_nodes(self) -> int:
        r"""Returns the number of nodes in the graph."""
        return self.x_action.size(0) + self.x_arm.size(0)
    
    @property
    def num_arm_features(self) -> int:
        return self.x_arm.size(1)
    
    @property
    def num_action_features(self) -> int:
        return self.x_action.size(1)
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_action.size(0)], [self.x_arm.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)
    
    def is_undirected(self):
        return NotImplementedError

