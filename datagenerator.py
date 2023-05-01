import torch
import torch.nn as nn
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import os
import json


def mod_x(x):
    return np.sum(np.abs(x), axis=1, keepdims=True)


def l1_norm(x):
    return np.sum(np.abs(x), axis=1, keepdims=True)


def square(x):
    return x**2


def combination_function(x):
    return np.max(np.hstack([mod_x(x), square(x)]), keepdims=True, axis=1)


def l2_norm(x):
    return np.sqrt(np.sum(x, axis=1, keepdims=True)**2)


def mod_function(x):
    return np.sum((3 / 4) * np.abs(x + 5) + (1 / 2) * np.abs(x - 5) +
                  (3 / 2) * np.abs(x - 10),
                  keepdims=True,
                  axis=1)


def mod_function2(x):
    return np.sum(np.abs(x + 3) + np.abs(x + 9) + np.abs(x + 5),
                  keepdims=True,
                  axis=1)


# def combination_function(x):
#     return np.max(np.hstack(
#         [mod_x(x), x**2, mod_function(x),
#          mod_function2(x)]),
#                   keepdims=True,
#                   axis=1)


def datageneration(args, function_dic):
    orig_function = function_dic[args.function]
    input_dim = args.data_dim
    save_dir = os.path.join(args.data_dir, args.function)

    if os.path.isdir(save_dir):
        if not args.force:
            print("Already existing data")
            exit()
    else:
        os.mkdir(save_dir)

    x = np.random.uniform(low=args.low,
                          high=args.high,
                          size=(args.num_points, input_dim))

    functional_output = orig_function(x)

    x = torch.from_numpy(x).to(torch.float32)
    functional_output = torch.from_numpy(functional_output).to(torch.float32)

    train_data, test_data, train_target, test_target = train_test_split(
        x, functional_output, test_size=args.test_ratio)

    np.save(os.path.join(save_dir, "train_data.npy"), train_data.cpu().numpy())
    np.save(os.path.join(save_dir, "train_target.npy"),
            train_target.cpu().numpy())
    np.save(os.path.join(save_dir, "test_data.npy"), test_data.cpu().numpy())
    np.save(os.path.join(save_dir, "test_target.npy"),
            test_target.cpu().numpy())

    train_dataset = TensorDataset(train_data, train_target)
    test_dataset = TensorDataset(test_data, test_target)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    torch.save(train_loader, os.path.join(save_dir, 'trainloader.pth'))
    torch.save(test_loader, os.path.join(save_dir, 'testloader.pth'))

    with open(os.path.join(save_dir, "config.json"), "w") as file:
        json.dump(vars(args), file)

    with open(os.path.join("run_configs", args.common_loc, "data_config.json"),
              "w") as file:
        json.dump(vars(args), file, indent=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="datagenerator.py")

    parser.add_argument("--num_points",
                        type=int,
                        default=10**5,
                        help="number of points in the range")
    parser.add_argument("--function",
                        type=str,
                        default="mod_x",
                        choices=[
                            "mod_x", "l1_norm", "l2_norm", "mod_function",
                            "combination_function"
                        ])
    parser.add_argument("--test_ratio",
                        type=float,
                        default=0.1,
                        help="train test split ratio")
    parser.add_argument("--data_dir",
                        type=str,
                        default="data",
                        help="parent data dir")
    parser.add_argument("--force",
                        type=bool,
                        default=False,
                        help="regenerate the data even if data already exists")
    parser.add_argument("--common_loc", type=str, required=True)
    parser.add_argument("--data_dim", type=int, default=1)
    parser.add_argument("--high", type=float, default=100, help="upper limit")
    parser.add_argument("--low", type=float, default=-100, help="lower limit")
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    # function, input_dimension, save_dir_name
    function_dic = {
        "mod_x": mod_x,
        "l1_norm": l1_norm,
        "l2_norm": l2_norm,
        "mod_function": mod_function,
        "combination_function": combination_function,
        "mod_function2": mod_function2
    }

    if not os.path.isdir("data"):
        os.mkdir("data")

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)

    if not os.path.isdir("run_configs"):
        os.mkdir("run_configs")

    if not os.path.isdir(os.path.join("run_configs", args.common_loc)):
        os.mkdir(os.path.join("run_configs", args.common_loc))

    datageneration(args, function_dic)