import torch
import torch.nn as nn
from model.mlp import MLP
import json
import os
import argparse
import numpy as np


def optimum_value(network_dir, iterations):
    network_args = json.load(open(os.path.join(network_dir, "config.json")))
    network = MLP(network_args["in_dim"], network_args["hidden_dims"])
    network.load_state_dict(torch.load(os.path.join(network_dir,
                                                    "network.pt")))

    for param in network.parameters():
        param.requires_grad = False

    x = torch.tensor(torch.randn(1,
                                 network_args["in_dim"])).requires_grad_(True)

    optimizer = torch.optim.AdamW([x], lr=1e-4)
    for idx in range(1, iterations + 1):
        optimizer.zero_grad()

        y = network(x)
        loss = y**2
        loss.backward()

        optimizer.step()

    return network(x).item(), x.detach().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="optimum_value.py")

    parser.add_argument("--network_dir", type=str, required=True)
    parser.add_argument("--common_loc", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=30000)
    args = parser.parse_args()

    args.network_dir = os.path.join("trained_model", args.network_dir)

    opt_value, x = optimum_value(args.network_dir, args.iterations)
    print("Optimum value: ", opt_value)
    print("Optimum x: ", x)

    opt_dict = {
        "optimum_value": opt_value,
        "optimum_x": x.tolist(),
        "distance": np.sum(np.abs(x.detach().cpu().numpy()))
    }
    with open(
            os.path.join("run_configs", args.common_loc,
                         "ann_optimumvalue_config.json"), "w") as file:
        json.dump(opt_dict, file, indent=0)