import argparse
import random
import signal
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from utils import get_optimizer, prepare_data

warnings.simplefilter("ignore")


parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--sweep", type=bool, default=False)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--weight_decay", type=float, default=None)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--optimizer", type=str, default=None)
parser.add_argument("--validation_seed", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--project_name", type=str, default=None)
parser.add_argument("--save_model", type=bool, default=False)
parser.add_argument("--model_name", type=str, default="model.pth")
parser.add_argument("--dataset_name", type=str, default="adult")
parser.add_argument("--hidden_size_1", type=int, default=16)
parser.add_argument("--hidden_size_2", type=int, default=16)


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


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LinearClassificationNet(nn.Module):
    """
    A fully-connected single-layer linear NN for classification.
    """

    def __init__(self, input_size=11, output_size=2):
        super(LinearClassificationNet, self).__init__()
        self.layer1 = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        x = self.layer1(x.float())
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
            "seed": args.seed,
            "validation_seed": args.validation_seed,
        },
    )
    return wandb_run


def eval_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    f1 = 0
    predictions = []
    true_values = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            predicted = outputs.argmax(dim=1, keepdim=True)
            total += target.size(0)
            correct += (predicted == target.view_as(predicted)).sum().item()
            predictions.extend(predicted.cpu().numpy().flatten())
            true_values.extend(target.cpu().numpy().flatten())
    acc = correct / total

    f1 = f1_score(true_values, predictions)

    return acc, f1


def get_model(args):
    if args.dataset_name == "diabetes":
        model = SimpleModel(44, 2)
    elif args.dataset_name == "pima":
        model = NeuralNetwork(
            8, 2, hidden_size_1=args.hidden_size_1, hidden_size_2=args.hidden_size_2
        )
    elif args.dataset_name == "breast_cancer":
        model = SimpleModel(15, 2)
    elif args.dataset_name == "adult":
        model = SimpleModel(111, 2)
    elif args.dataset_name == "dutch":
        model = SimpleModel(11, 2)
    else:
        raise ValueError("Invalid dataset name")
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

        train_acc, train_f1 = eval_model(model, train_loader, device)
        wandb_run.log(
            {"train_accuracy": train_acc, "epoch": epoch, "train_f1": train_f1}
        )

        if args.sweep and val_loader is not None:
            val_acc, val_f1 = eval_model(model, val_loader, device)
            wandb_run.log(
                {"validation_accuracy": val_acc, "epoch": epoch, "val_f1": val_f1}
            )

        if test_loader is not None and not args.sweep:
            test_acc, test_f1 = eval_model(model, test_loader, device)
            wandb_run.log(
                {"test_accuracy": test_acc, "epoch": epoch, "test_f1": test_f1}
            )
    return model


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

    current_script_path = Path(__file__).resolve().parent
    (train_loader, val_loader, test_loader) = prepare_data(
        args=args, current_path=current_script_path
    )

    # Don't remove the seed setting
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = get_model(args)

    optimizer = get_optimizer(args.optimizer, model, args.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model = train_model(
        model,
        optimizer,
        train_loader,
        args.epochs,
        val_loader,
        test_loader,
        args,
        device,
    )

    # if not args.sweep:
    test_acc, f1 = eval_model(model, test_loader, device)
    print(f"Test accuracy: {test_acc}")
    print(f"Test f1: {f1}")
    y = test_loader.dataset.tensors[1]
    print(f"Majority Classifier Accuracy: {max(y.mean(), 1 - y.mean())}")

    if args.save_model:
        torch.save(
            model, f"../../artifacts/{args.dataset_name}/bb/{args.model_name}.pth"
        )
