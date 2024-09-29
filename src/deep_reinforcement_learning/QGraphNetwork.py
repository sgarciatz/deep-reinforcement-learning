import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces.graph import GraphInstance


class QGraphNetwork(nn.Module):

    def __init__(self,
                 node_observations: int,
                 edge_observations: int,
                 graph_convolutions: int,
                 actions: int,
                 device: str = "cpu"):
        super(QGraphNetwork, self).__init__()

        self.node_observations: int = node_observations
        self.edge_observations: int = edge_observations
        self.observations = self.node_observations + self.edge_observations
        self.graph_convolutions: int = graph_convolutions
        self.n_actions: int = actions
        self.device = device
        self.epsilon = torch.zeros(graph_convolutions).to(device=device)
        self.mlps: list[nn.Sequential] = []
        for _ in range(self.graph_convolutions):
            mlp: nn.Sequential = nn.Sequential(
                nn.Linear(self.node_observations, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, self.node_observations)
            ).to(device=self.device)
            self.mlps.append(mlp)
        self.mlps = nn.ModuleList(self.mlps)
        self.readout_mlp = nn.Linear(1, self.n_actions)


    def forward(self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        node_features = x[0].clone().detach().repeat((1,1,1)).to(self.device)
        edge_features = x[1].clone().detach().repeat((1,1,1)).to(self.device)
        adj_matrix = x[2].clone().detach().repeat((1,1,1)).to(self.device)
        # base_case
        node_embeddings = node_features
        k_node_embeddings = torch.zeros_like(node_embeddings).repeat(
            self.graph_convolutions+1,
            1,
            1,
            1)
        k_node_embeddings[0] = node_embeddings
        for i in range(self.graph_convolutions):
            # aggregate neighbour messages
            neighbour_agg = torch.matmul(adj_matrix,
                                         node_embeddings).to(self.device)
            # update node values
            node_embeddings = ((1 + self.epsilon[i])\
                * node_embeddings\
                + neighbour_agg).to(self.device)
            # apply layer mlp
            node_embeddings = self.mlps[i](node_embeddings)
            k_node_embeddings[i+1] = node_embeddings
        # readout function
        # transposed to -> batch, k (graph_layers), nodes, features
        k_node_embeddings = k_node_embeddings.transpose(0, 1)
        graph_embedding = k_node_embeddings.sum(dim=1).sum(dim=2).sum(dim=1).repeat((1,1)).T
        graph_embedding = self.readout_mlp(graph_embedding)
        return graph_embedding

# torch.manual_seed(0)

# node_observations = 2
# edge_observations = 1
# graph_convolutions = 2
# actions = 4

# qgn = QGraphNetwork(node_observations,
#                     edge_observations,
#                     graph_convolutions,
#                     actions,
#                     "cpu")

# node_features = torch.tensor([[10., 1.],
#                               [20., 2.],
#                               [30., 3.],
#                               [40., 4.]])


# node_features = node_features.repeat(1, 1, 1)

# # for i in range(1, 5):
# #     node_features[i] *= (i+1)

# edge_features = torch.tensor([[10.],
#                               [20.],
#                               [30.],
#                               [40.]])


# adj_matrix = torch.tensor([[1., 0., 0., 1.],
#                            [0., 1., 1., 0.],
#                            [0., 1., 1., 0.],
#                            [1., 0., 0., 1.]])

# print("Adj matrix")
# print(torch.matmul(adj_matrix, adj_matrix))
# print("Node features")
# print(node_features)
# print("---------------------------------------------------------------")

# x = (node_features, edge_features, adj_matrix)
# q_predictions = qgn(x)
# print(q_predictions)