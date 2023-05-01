from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import json
import sys
from rbf import MyDataset
from torch.utils.data import Dataset, DataLoader

from datagenerator import mod_x, combination_function, mod_function, mod_function2, l1_norm, l2_norm


class PolynomialModel(nn.Module):
    def __init__(self, degree):
        super().__init__()
        self._degree = degree
        self.linear = nn.Linear(self._degree + 1, 1)

    def forward(self, x):
        return self.linear(self._polynomial_features(x))

    def _polynomial_features(self, x):
        return torch.hstack([x**i for i in range(0, self._degree + 1)])


def train(model, args, train_data, train_target, test_data, test_target,
          loss_func):
    model.train()
    obs = train_data.size(0)
    trainset = MyDataset(train_data, train_target)
    trainloader = DataLoader(trainset,
                             batch_size=args.batch_size,
                             shuffle=True)
    optimiser = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    epoch = 0
    while epoch < args.epochs:
        epoch += 1
        current_loss = 0
        batches = 0
        progress = 0
        for x_batch, y_batch in trainloader:
            batches += 1
            optimiser.zero_grad()
            y_hat = model(x_batch)
            loss = loss_func(y_hat, y_batch)
            current_loss += (1 / batches) * (loss.item() - current_loss)
            loss.backward()
            optimiser.step()
            progress += y_batch.size(0)
            sys.stdout.write('\rEpoch: %d, Progress: %d/%d, Loss: %f      ' % \
                                (epoch, progress, obs, current_loss))
            sys.stdout.flush()

        if epoch % args.evaluation_period == 0:
            model.eval()
            preds = model.forward(test_data)
            test_loss = loss_func(preds, test_target)
            print('Test loss: %f' % test_loss)

            MAE_loss = torch.mean(torch.abs(preds - test_target))
            if MAE_loss < args.epsilon:
                print('Early Stopping')
                break
            model.train()

    return model


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

    model = PolynomialModel(args.degree).to(args.device)

    model = train(model, args, train_data, train_target, test_data,
                  test_target, nn.MSELoss())

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    opt_x = torch.tensor(torch.randn(1, args.in_dim)).to(
        args.device).requires_grad_(True)

    optimizer = torch.optim.AdamW([opt_x], lr=1e-4)
    for idx in range(1, args.iterations + 1):
        optimizer.zero_grad()

        y = model(opt_x)
        loss = y**2
        loss.backward()

        optimizer.step()

    opt_value = model(opt_x).item()
    opt_x = opt_x.detach().cpu().numpy()

    print("Optimum value: ", opt_value)
    print("Optimum x: ", opt_x)
    opt_dict = {"optimum_value": opt_value, "optimum_x": opt_x.tolist()}
    with open(
            os.path.join("run_configs", args.common_loc,
                         "poly_reg_optimumvalue_config.json"), "w") as file:
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
    parser.add_argument("--degree", type=int, default=2)

    args = parser.parse_args()
    args.data_dir = os.path.join("data", args.data_dir)

    with open(os.path.join(args.data_dir, "config.json")) as file:
        data_arguments = json.load(file)

    args.in_dim = data_arguments["data_dim"]

    main(args)