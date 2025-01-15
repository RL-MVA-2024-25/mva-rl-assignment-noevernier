import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, n_action, nb_neurons, depth):
        super(DQN, self).__init__()
        layers = []
        
        layers.append(nn.Linear(state_dim, nb_neurons))
        layers.append(nn.ReLU())

        for _ in range(depth):
            layers.append(nn.Linear(nb_neurons, nb_neurons))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(nb_neurons, n_action))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)