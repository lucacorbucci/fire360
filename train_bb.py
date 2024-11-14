import argparse
import random
import signal
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from tqdm.auto import tqdm

from utils import get_optimizer, prepare_data

warnings.simplefilter("ignore")


parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--sweep", type=bool, default=False)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--optimizer", type=str, default=None)
parser.add_argument("--validation_seed", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--project_name", type=str, default=None)
parser.add_argument("--save_model", type=bool, default=False)
parser.add_argument("--model_name", type=str, default="model.pth")
parser.add_argument("--dataset_name", type=str, default="adult")

def signal_handler(sig, frame):
    print("Gracefully stopping your experiment! Keep calm!")
    global wandb_run
    if wandb_run:
        wandb_run.finish()
    sys.exit(0)

class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 32)
        self.layer2 = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x.float()))
        x = self.layer2(x)
        return x


def setup_wandb(args):
    wandb_run = wandb.init(
        # set the wandb project where this run will be logged
        project=args.project_name,
        dir="/raid/lcorbucci/wandb_tmp",
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "optimizer": args.optimizer,
        },
    )
    return wandb_run


def eval_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            predicted = outputs.argmax(dim=1, keepdim=True)
            total += target.size(0)
            correct += (predicted == target.view_as(predicted)).sum().item()
    acc = correct / total
    return acc

def get_model(args):
    if args.dataset_name == "diabetes":
        model = SimpleModel(44, 2)
    elif args.dataset_name == "pima":
        model = SimpleModel(8, 2)
    elif args.dataset_name == "breast_cancer":
        model = SimpleModel(15, 2)
    return model

def train_model(
    model,
    optimizer,
    train_loader,
    epochs,
    val_loader=None,
    test_loader=None,
    args=None,
    device=None,
):
    wandb_run = setup_wandb(args)

    for epoch in tqdm(range(epochs)):
        model.train()
        for data, labels in train_loader:
            labels = labels.type(torch.LongTensor)
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb_run.log({"train_loss": loss.item()})

        train_acc = eval_model(model, train_loader, device)
        wandb_run.log({"train_accuracy": train_acc, "epoch": epoch})

        if args.sweep and val_loader is not None:
            val_acc = eval_model(model, val_loader, device)
            wandb_run.log({"validation_accuracy": val_acc, "epoch": epoch})

        if test_loader is not None and not args.sweep:
            test_acc = eval_model(model, test_loader, device)
            wandb_run.log({"test_accuracy": test_acc, "epoch": epoch})


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    args = parser.parse_args()

    
    # Don't remove the seed setting
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.validation_seed is None:
        validation_seed = int(str(time.time()).split(".")[1]) * args.seed
        args.validation_seed = validation_seed
    
    (
        train_loader,
        val_loader,
        test_loader
    ) = prepare_data(args)

    # Don't remove the seed setting
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = get_model(args)

    optimizer = get_optimizer(args.optimizer, model, args.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_model(
        model,
        optimizer,
        train_loader,
        args.epochs,
        val_loader,
        test_loader,
        args,
        device,
    )

    if args.save_model:
        # save model with and without state dict
        torch.save(
            model, f"../../artifacts/{args.dataset_name}/bb/{args.model_name}.pth"
        )
