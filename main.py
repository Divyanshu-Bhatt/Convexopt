import torch
import numpy as np
import argparse
import os
import json

from train import model_initialisation, model_train


def main(args):
    train_loader = torch.load(os.path.join(args.data_dir, "trainloader.pth"))
    test_loader = torch.load(os.path.join(args.data_dir, "testloader.pth"))

    network_credentials = model_initialisation(args)
    model_train(network_credentials, train_loader, test_loader, args)

    with open(os.path.join(args.model_save_dir, "config.json"), "w") as file:
        json.dump(vars(args), file)

    with open(os.path.join(args.common_loc, "ann_model_config.json"),
              "w") as file:
        json.dump(vars(args), file, indent=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main.py")

    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data_dir", type=str, default="mod_x")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--evaluation_period", type=int, default=10)
    parser.add_argument("--model_save_dir", type=str, required=True)
    parser.add_argument("--common_loc", type=str, required=True)
    parser.add_argument("--weight_decay",
                        type=float,
                        default=1e-4,
                        help="regulariser")

    args = parser.parse_args()

    args.data_dir = os.path.join("data", args.data_dir)
    args.model_save_dir = os.path.join("trained_model", args.model_save_dir)
    args.hidden_dims = [64, 32]

    if not os.path.isdir("trained_model"):
        os.mkdir("trained_model")

    if os.path.isdir(args.model_save_dir):
        print("model folder already exist")
        exit()
    os.mkdir(args.model_save_dir)

    args.common_loc = os.path.join("run_configs", args.common_loc)
    if not os.path.isdir(args.common_loc):
        os.mkdir(args.common_loc)

    with open(os.path.join(args.data_dir, "config.json")) as file:
        data_arguments = json.load(file)

    args.in_dim = data_arguments["data_dim"]

    main(args)