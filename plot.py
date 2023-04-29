import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from model.mlp import MLP


def plot():

    return None


if __name__ == "__main__":

    data_dir = os.path.join('data', 'combination_function')
    dataloader = torch.load(os.path.join(data_dir, 'trainloader.pth'))

    train_values = []
    train_labels = []

    for data, label in dataloader:
        train_values.append(data.numpy())
        train_labels.append(label.numpy())

    train_values = np.vstack(train_values)
    train_labels = np.vstack(train_labels)

    plt.scatter(train_values, train_labels)

    model_dir = os.path.join('trained_model', 'MLP_205', 'network.pt')
    data = np.arange(-100, 100, 0.1).reshape(-1, 1)

    model = MLP(in_dim=1)
    model.load_state_dict(torch.load(model_dir))

    predictions = []

    for i in range(len(data)):
        d = torch.from_numpy(data[i]).to(torch.float32)
        output = model(d).detach().numpy()
        predictions.append(output)

    predictions = np.array(predictions).reshape(-1, 1)

    plt.scatter(data, predictions)
    plt.ylim([0, 100])
    plt.xlim([-10, 10])
    plt.show()

    # plot(dataloader)