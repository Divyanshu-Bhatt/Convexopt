import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims=[256, 128]):
        super(MLP, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(in_dim, hidden_dims[0]), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(hidden_dims[1], 1))

    def forward(self, in_features):
        x = self.fc1(in_features)
        x = self.fc2(x)
        return self.fc3(x)