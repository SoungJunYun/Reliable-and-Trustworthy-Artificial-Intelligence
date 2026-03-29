import random
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from tqdm.auto import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def fit_model(model, train_loader, test_loader, device, epochs=3, lr=1e-3, save_path=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    best_acc = 0.0
    best_state = deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                train_loss=f"{running_loss / total:.4f}",
                train_acc=f"{100.0 * correct / total:.2f}%"
            )

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        val_loss, val_acc = evaluate(model, test_loader, device)

        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    return model, best_acc