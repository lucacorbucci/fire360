import argparse
import random
import signal
import sys
import time
import warnings
from pathlib import Path
from types import FrameType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from fire360.bb_architectures import MultiClassModel, SimpleModel
from fire360.utils import get_optimizer, prepare_data
from loguru import logger
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

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
parser.add_argument("--shuffle_seed", type=int, default=16)


def signal_handler(_sig: int, frame: FrameType | None) -> None:
    """
    Function used to handle the SIGINT signal
    """
    logger.info("Gracefully stopping your experiment! Keep calm!")
    global wandb_run
    if wandb_run:
        wandb_run.finish()
    sys.exit(0)


def setup_wandb(args: argparse.Namespace) -> wandb.sdk.wandb_run.Run:
    wandb_run = wandb.init(
        # set the wandb project where this run will be logged
        project=args.project_name,
        dir="/raid/lcorbucci/wandb_tmp",
        name=f"{args.dataset_name}",
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "optimizer": args.optimizer,
            "seed": args.seed,
            "validation_seed": args.validation_seed,
            "dataset_name": args.dataset_name,
        },
    )
    return wandb_run


def eval_model(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[float, float]:
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

    f1 = f1_score(true_values, predictions, average="weighted")

    return acc, f1


def get_model(
    args: argparse.Namespace,
) -> nn.Module:
    match args.dataset_name:
        case "diabetes":
            model = SimpleModel(44, 2)
        case "breast_cancer":
            model = SimpleModel(15, 2)
        case "adult":
            model = SimpleModel(111, 2)
        case "dutch":
            model = SimpleModel(11, 2)
        case "shuttle":
            model = MultiClassModel(9, 7)
        case "covertype":
            model = MultiClassModel(54, 7)
        case "letter":
            model = MultiClassModel(16, 26)
        case "house16":
            model = SimpleModel(16, 2)
        case _:
            raise ValueError("Invalid dataset name")
    return model


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    epochs: int,
    args: argparse.Namespace,
    wandb_run: wandb.sdk.wandb_run.Run,
    val_loader: torch.utils.data.DataLoader = None,
    test_loader: torch.utils.data.DataLoader = None,
    device: torch.device = None,
) -> nn.Module:
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
        wandb_run.log({"train_accuracy": train_acc, "epoch": epoch, "train_f1": train_f1})

        if args.sweep and val_loader is not None:
            val_acc, val_f1 = eval_model(model, val_loader, device)
            wandb_run.log({"validation_accuracy": val_acc, "epoch": epoch, "val_f1": val_f1})

        if test_loader is not None and not args.sweep:
            test_acc, test_f1 = eval_model(model, test_loader, device)
            wandb_run.log({"test_accuracy": test_acc, "epoch": epoch, "test_f1": test_f1})
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
    (train_loader, val_loader, test_loader) = prepare_data(args=args, current_path=current_script_path)

    # I wanted to add this for testing purposes. In this way I can shuffle the
    # dataset before running the training. This is useful to check if the model
    # is overfitting or not and to have multiple runs with different seeds.
    if args.sweep is False:
        logger.info("Shuffling the train_loader")
        np.random.seed(args.shuffle_seed)
        random.seed(args.shuffle_seed)
        torch.manual_seed(args.shuffle_seed)
        torch.cuda.manual_seed_all(args.shuffle_seed)

        dataset = train_loader.dataset
        # shuffle the dataset
        indices = np.arange(len(dataset))
        rng = np.random.default_rng(args.shuffle_seed)
        rng.shuffle(indices)
        dataset.tensors = (dataset.tensors[0][indices], dataset.tensors[1][indices])

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    # Don't remove the seed setting
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = get_model(args)

    optimizer = get_optimizer(args.optimizer, model, args.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    wandb_run = setup_wandb(args)

    model = train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        epochs=args.epochs,
        val_loader=val_loader,
        test_loader=test_loader,
        args=args,
        wandb_run=wandb_run,
        device=device,
    )

    # if not args.sweep:
    test_acc, f1 = eval_model(model, test_loader, device)
    logger.info(f"Test accuracy: {test_acc}")
    logger.info(f"Test f1: {f1}")
    y = test_loader.dataset.tensors[1]
    logger.info(f"Majority Classifier Accuracy: {max(y.mean(), 1 - y.mean())}")

    if args.save_model:
        torch.save(model, current_script_path / f"../../../artifacts/{args.dataset_name}/bb/{args.model_name}.pth")
