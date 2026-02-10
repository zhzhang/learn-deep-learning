from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from modules.torch.reference import MLP

try:
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    from torchvision.transforms.functional import to_pil_image
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "torchvision is required for Fashion-MNIST training. Install it first."
    ) from exc

PROGRESS_PREFIX = "PROGRESS:"
SAMPLES_PER_CLASS = 10
FASHION_MNIST_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def _emit_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    event: dict[str, Any],
) -> None:
    print(f"{PROGRESS_PREFIX}{json.dumps(event)}", flush=True)
    if progress_callback is not None:
        progress_callback(event)


def _load_fmnist_dataset(
    data_dir: str | Path = "data",
    *,
    train: bool,
) -> datasets.FashionMNIST:
    return datasets.FashionMNIST(
        root=str(data_dir),
        train=train,
        download=True,
        transform=ToTensor(),
    )


def _build_balanced_sample_descriptors(
    dataset: datasets.FashionMNIST,
    *,
    samples_per_class: int = SAMPLES_PER_CLASS,
) -> list[dict[str, Any]]:
    class_names = list(dataset.classes)
    class_count = len(class_names)
    counts = [0] * class_count
    descriptors: list[dict[str, Any]] = []

    for sample_index in range(len(dataset)):
        class_index = int(dataset.targets[sample_index])
        if class_index < 0 or class_index >= class_count:
            continue
        slot = counts[class_index]
        if slot >= samples_per_class:
            continue

        sample_id = class_index * samples_per_class + slot
        descriptors.append(
            {
                "sample_id": sample_id,
                "sample_index": sample_index,
                "class_index": class_index,
                "class_name": class_names[class_index],
                "slot": slot,
            }
        )
        counts[class_index] += 1

        if all(count >= samples_per_class for count in counts):
            break

    descriptors.sort(key=lambda item: (int(item["class_index"]), int(item["slot"])))
    return descriptors


def get_balanced_test_samples(
    data_dir: str | Path = "data",
    *,
    samples_per_class: int = SAMPLES_PER_CLASS,
) -> list[dict[str, Any]]:
    test_data = _load_fmnist_dataset(data_dir=data_dir, train=False)
    return _build_balanced_sample_descriptors(
        test_data, samples_per_class=samples_per_class
    )


def get_fmnist_sample_png(sample_id: int, data_dir: str | Path = "data") -> bytes:
    if sample_id < 0:
        raise ValueError("Sample id must be non-negative.")
    test_data = _load_fmnist_dataset(data_dir=data_dir, train=False)
    samples = _build_balanced_sample_descriptors(test_data)
    sample = next((item for item in samples if int(item["sample_id"]) == sample_id), None)
    if sample is None:
        raise IndexError("Sample id is out of range.")
    image_tensor, _ = test_data[int(sample["sample_index"])]
    image = to_pil_image(image_tensor)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def evaluate_samples(
    model: nn.Module,
    test_data: datasets.FashionMNIST,
    samples: list[dict[str, Any]],
    device: str,
) -> list[dict[str, Any]]:
    sample_indices = [int(sample["sample_index"]) for sample in samples]
    images = torch.stack([test_data[index][0] for index in sample_indices]).to(device)
    labels = torch.tensor(
        [int(test_data[index][1]) for index in sample_indices], device=device
    )

    model.eval()
    with torch.no_grad():
        predictions = model(images).argmax(1)

    results: list[dict[str, Any]] = []
    for sample, expected, predicted in zip(samples, labels, predictions):
        expected_label = int(expected.item())
        predicted_label = int(predicted.item())
        results.append(
            {
                "sample_id": int(sample["sample_id"]),
                "class_index": int(sample["class_index"]),
                "class_name": str(sample["class_name"]),
                "slot": int(sample["slot"]),
                "expected_label": expected_label,
                "predicted_label": predicted_label,
                "predicted_class_name": FASHION_MNIST_CLASS_NAMES[predicted_label],
                "correct": expected_label == predicted_label,
            }
        )
    return results


def train_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    epochs: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    log_interval: int = 1,
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
    data_dir: str | Path = "data",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    # Re-seed per invocation so each "Start Training" begins from a clean baseline.
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

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

    sample_descriptors = _build_balanced_sample_descriptors(
        test_data, samples_per_class=SAMPLES_PER_CLASS
    )
    _emit_progress(
        progress_callback,
        {
            "stage": "samples_ready",
            "samples": sample_descriptors,
        },
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
        sample_results = evaluate_samples(model, test_data, sample_descriptors, device)
        _emit_progress(
            progress_callback,
            {"stage": "sample_eval", "epoch": epoch, "samples": sample_results},
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

    _emit_progress(
        progress_callback,
        {"stage": "completed", "epochs": epochs},
    )
    return {
        "status": "completed",
        "device": device,
        "epochs": epochs,
        "history": history,
    }


if __name__ == "__main__":
    train_reference_model()
