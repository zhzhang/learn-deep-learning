from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from modules.torch.reference import MLP

try:
    from torchvision import datasets
    from torchvision.transforms import ToTensor
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "torchvision is required for Fashion-MNIST training. Install it first."
    ) from exc

PROGRESS_PREFIX = "PROGRESS:"
DEFAULT_MODEL_OUTPUT = Path(__file__).resolve().parent / "reference_model.pth"


def _emit_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    event: dict[str, Any],
) -> None:
    print(f"{PROGRESS_PREFIX}{json.dumps(event)}", flush=True)
    if progress_callback is not None:
        progress_callback(event)


def train_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    epochs: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    log_interval: int = 100,
) -> dict[str, float]:
    model.train()
    dataset_size = len(dataloader.dataset)
    running_loss = 0.0
    correct = 0
    seen = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_size = y.size(0)
        seen += batch_size
        running_loss += loss.item() * batch_size
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch_idx % log_interval == 0:
            current = min((batch_idx + 1) * batch_size, dataset_size)
            _emit_progress(
                progress_callback,
                {
                    "stage": "train",
                    "epoch": epoch,
                    "epochs": epochs,
                    "batch": batch_idx + 1,
                    "batches": len(dataloader),
                    "loss": loss.item(),
                    "samples_seen": current,
                    "samples_total": dataset_size,
                },
            )

    average_loss = running_loss / max(1, seen)
    accuracy = correct / max(1, seen)
    summary = {
        "stage": "train_summary",
        "epoch": epoch,
        "epochs": epochs,
        "loss": average_loss,
        "accuracy": accuracy,
    }
    _emit_progress(progress_callback, summary)
    return {"loss": average_loss, "accuracy": accuracy}


def test_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str,
    epoch: int,
    epochs: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, float]:
    model.eval()
    dataset_size = len(dataloader.dataset)
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += loss_fn(pred, y).item() * y.size(0)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_loss = total_loss / max(1, dataset_size)
    accuracy = correct / max(1, dataset_size)
    summary = {
        "stage": "test_summary",
        "epoch": epoch,
        "epochs": epochs,
        "loss": average_loss,
        "accuracy": accuracy,
    }
    _emit_progress(progress_callback, summary)
    return {"loss": average_loss, "accuracy": accuracy}


def train_reference_model(
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    output_path: str | Path = DEFAULT_MODEL_OUTPUT,
    data_dir: str | Path = "data",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _emit_progress(progress_callback, {"stage": "setup", "device": device})

    training_data = datasets.FashionMNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.FashionMNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model = MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            dataloader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=epochs,
            progress_callback=progress_callback,
        )
        test_metrics = test_epoch(
            dataloader=test_dataloader,
            model=model,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            epochs=epochs,
            progress_callback=progress_callback,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    _emit_progress(
        progress_callback,
        {"stage": "completed", "model_path": str(output.resolve()), "epochs": epochs},
    )
    return {
        "status": "completed",
        "device": device,
        "epochs": epochs,
        "model_path": str(output.resolve()),
        "history": history,
    }


if __name__ == "__main__":
    train_reference_model()
