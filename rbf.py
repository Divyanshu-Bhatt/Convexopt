import torch_rbf as rbf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json

from datagenerator import mod_x, combination_function, mod_function, mod_function2, l1_norm, l2_norm


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return (x, y)


class Network(nn.Module):
    def __init__(self, layer_widths, layer_centres, basis_func):
        super(Network, self).__init__()
        self.rbf_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for i in range(len(layer_widths) - 1):
            self.rbf_layers.append(
                rbf.RBF(layer_widths[i], layer_centres[i], basis_func))
            self.linear_layers.append(
                nn.Linear(layer_centres[i], layer_widths[i + 1]))

    def forward(self, x):
        out = x
        for i in range(len(self.rbf_layers)):
            out = self.rbf_layers[i](out)
            out = self.linear_layers[i](out)
        return out

    def fit(self, train_data, train_target, test_data, test_target, epochs,
            batch_size, lr, loss_func, weight_decay, evaluation_period,
            epsilon):

        self.train()
        obs = train_data.size(0)
        trainset = MyDataset(train_data, train_target)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        optimiser = torch.optim.AdamW(self.parameters(),
                                      lr=lr,
                                      weight_decay=weight_decay)
        epoch = 0
        while epoch < epochs:
            epoch += 1
            current_loss = 0
            batches = 0
            progress = 0
            for x_batch, y_batch in trainloader:
                batches += 1
                optimiser.zero_grad()
                y_hat = self.forward(x_batch)
                loss = loss_func(y_hat, y_batch)
                current_loss += (1 / batches) * (loss.item() - current_loss)
                loss.backward()
                optimiser.step()
                progress += y_batch.size(0)
                sys.stdout.write('\rEpoch: %d, Progress: %d/%d, Loss: %f      ' % \
                                 (epoch, progress, obs, current_loss))
                sys.stdout.flush()

            if epoch % evaluation_period == 0:
                self.eval()
                preds = self.forward(test_data)
                test_loss = loss_func(preds, test_target)
                print('Test loss: %f' % test_loss)

                MAE_loss = torch.mean(torch.abs(preds - test_target))
                if MAE_loss < epsilon:
                    print('Early Stopping')
                    break
                self.train()


def main(args):
    train_data = torch.from_numpy(
        np.load(os.path.join(args.data_dir, "train_data.npy"))).to(
            torch.float32).to(args.device)
    train_target = torch.from_numpy(
        np.load(os.path.join(args.data_dir, "train_target.npy"))).to(
            torch.float32).to(args.device)
    test_data = torch.from_numpy(
        np.load(os.path.join(args.data_dir, "test_data.npy"))).to(
            torch.float32).to(args.device)
    test_target = torch.from_numpy(
        np.load(os.path.join(args.data_dir, "test_target.npy"))).to(
            torch.float32).to(args.device)

    layer_widths = [2, 1]
    layer_centres = [40]

    rbfnet = Network(layer_widths, layer_centres, rbf.gaussian).to(args.device)

    rbfnet.fit(train_data, train_target,
               test_data, test_target, args.epochs, args.batch_size, args.lr,
               nn.MSELoss(), args.weight_decay, args.evaluation_period,
               args.epsilon)

    rbfnet.eval()
    for param in rbfnet.parameters():
        param.requires_grad = False

    opt_x = torch.tensor(torch.randn(1, args.in_dim)).to(
        args.device).requires_grad_(True)

    optimizer = torch.optim.AdamW([opt_x], lr=1e-4)
    for idx in range(1, args.iterations + 1):
        optimizer.zero_grad()

        y = rbfnet(opt_x)
        loss = y**2
        loss.backward()

        optimizer.step()

    opt_value = rbfnet(opt_x).item()
    opt_x = opt_x.detach().cpu().numpy()

    print("Optimum value: ", opt_value)
    print("Optimum x: ", opt_x)
    opt_dict = {"optimum_value": opt_value, "optimum_x": opt_x.tolist()}
    with open(
            os.path.join("run_configs", args.common_loc,
                         "rbf_optimumvalue_config.json"), "w") as file:
        json.dump(opt_dict, file, indent=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--common_loc", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="mod_x")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--evaluation_period", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    args.data_dir = os.path.join("data", args.data_dir)

    with open(os.path.join(args.data_dir, "config.json")) as file:
        data_arguments = json.load(file)

    args.in_dim = data_arguments["data_dim"]

    main(args)