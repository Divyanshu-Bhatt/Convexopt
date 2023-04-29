import numpy as np
import torch
import torch.nn as nn
import os

from model.mlp import MLP


def model_initialisation(args):
    network = MLP(args.in_dim, args.hidden_dims)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(network.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[20, 200],
                                                     gamma=0.1)

    return (network, criterion, scheduler, optimizer)


def model_train(network_credentials, train_loader, val_loader, test_loader,
                args):
    network, criterion, schedular, optimizer = network_credentials
    del network_credentials

    network.to(args.device)
    args.best_val_loss = np.inf
    args.best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        network.train()
        MLE_loss = 0
        counter = 0

        for input_embeds, target in train_loader:
            optimizer.zero_grad()

            input_embeds = input_embeds.to(args.device)
            target = target.to(args.device)

            prediction = network(input_embeds)

            loss = criterion(prediction, target)

            loss.backward()
            optimizer.step()

            MLE_loss += loss
            counter += 1
        schedular.step()

        print(f"\nEpoch: {epoch}")
        print(f"Training Loss| MLE Loss: {MLE_loss/counter}")

        if epoch % args.evaluation_period == 0:
            print("Validation Loss")
            _, val_loss = model_eval(network, val_loader, criterion, args)

            if val_loss < args.best_val_loss:
                torch.save(network.state_dict(),
                           os.path.join(args.model_save_dir, "network.pt"))
                args.best_val_loss = val_loss
                args.best_epoch = epoch
                print(f"Saving model at Epoch: {epoch}")

    network.load_state_dict(
        torch.load(os.path.join(args.model_save_dir, "network.pt")))

    print("\nTest Loss")
    _, args.test_loss = model_eval(network, test_loader, criterion, args)


def model_eval(network, dataloader, criterion, args):

    network.eval()
    network.to(args.device)

    MLE_loss = 0
    counter = 0
    predictions = []

    with torch.no_grad():
        for input_embeds, target in dataloader:
            input_embeds = input_embeds.to(args.device)
            target = target.to(args.device)

            prediction = network(input_embeds)
            loss = criterion(prediction, target)

            MLE_loss += loss

            predictions.append(prediction.detach().cpu().numpy())
            counter += 1

    print(f"MLE Loss: {MLE_loss/counter}")

    return np.vstack(predictions), (MLE_loss / counter).item()
