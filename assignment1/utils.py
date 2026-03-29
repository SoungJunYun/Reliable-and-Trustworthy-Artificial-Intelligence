import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_mnist_loaders(batch_size=128, data_root="./data"):
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def get_cifar10_loaders(batch_size=64, data_root="./data", image_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # normalize는 모델 내부에서
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    class_names = train_dataset.classes
    return train_loader, test_loader, class_names


def get_target_labels(labels, num_classes=10):
    return (labels + 1) % num_classes


def label_to_name(label, class_names=None):
    if class_names is None:
        return str(label)
    return str(class_names[label])


def save_attack_triplet(x, x_adv, pred_orig, pred_adv, save_path, class_names=None):
    x = x.detach().cpu()
    x_adv = x_adv.detach().cpu()
    perturb = x_adv - x

    c = x.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    if c == 1:
        axes[0].imshow(x.squeeze(0).numpy(), cmap="gray")
        axes[1].imshow(x_adv.squeeze(0).numpy(), cmap="gray")
        axes[2].imshow((perturb.squeeze(0).numpy() * 10.0), cmap="gray")
    else:
        axes[0].imshow(np.clip(x.permute(1, 2, 0).numpy(), 0, 1))
        axes[1].imshow(np.clip(x_adv.permute(1, 2, 0).numpy(), 0, 1))
        axes[2].imshow(np.clip(perturb.permute(1, 2, 0).numpy() * 10.0 + 0.5, 0, 1))

    axes[0].set_title(f"Original\nPred: {label_to_name(pred_orig, class_names)}")
    axes[1].set_title(f"Adversarial\nPred: {label_to_name(pred_adv, class_names)}")
    axes[2].set_title("Perturbation x10")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_csv(rows, save_path):
    if not rows:
        return

    keys = rows[0].keys()
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_attack(
    model,
    loader,
    attack_fn,
    dataset_name,
    attack_name,
    eps,
    device,
    results_dir,
    targeted=False,
    attack_kwargs=None,
    class_names=None,
    num_classes=10,
    num_samples=100,
    save_examples=False,
):
    """
    의미 있는 평가를 위해 clean input에서 원래 맞춘 샘플만 대상으로 삼음.
    """
    if attack_kwargs is None:
        attack_kwargs = {}

    model.eval()

    success = 0
    total = 0

    success_examples = []
    fallback_examples = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        for i in range(batch_size):
            if total >= num_samples:
                break

            x = images[i:i+1]
            y = labels[i:i+1]

            with torch.no_grad():
                orig_pred = model(x).argmax(dim=1)

            # 원래부터 틀린 샘플은 제외
            if orig_pred.item() != y.item():
                continue

            if targeted:
                target = get_target_labels(y, num_classes=num_classes).to(device)
                x_adv = attack_fn(model, x, target=target, eps=eps, **attack_kwargs)
                with torch.no_grad():
                    adv_pred = model(x_adv).argmax(dim=1)
                is_success = (adv_pred.item() == target.item())
            else:
                x_adv = attack_fn(model, x, label=y, eps=eps, **attack_kwargs)
                with torch.no_grad():
                    adv_pred = model(x_adv).argmax(dim=1)
                is_success = (adv_pred.item() != y.item())

            total += 1
            if is_success:
                success += 1

            example = (x.squeeze(0).cpu(), x_adv.squeeze(0).cpu(), orig_pred.item(), adv_pred.item())
            if is_success and len(success_examples) < 5:
                success_examples.append(example)
            elif len(fallback_examples) < 5:
                fallback_examples.append(example)

        if total >= num_samples:
            break

    if total == 0:
        raise RuntimeError(f"No correctly classified samples found for {dataset_name} / {attack_name}")

    success_rate = 100.0 * success / total

    if save_examples:
        chosen = success_examples[:]
        if len(chosen) < 5:
            chosen.extend(fallback_examples[: 5 - len(chosen)])

        for idx, (x_img, x_adv_img, pred_o, pred_a) in enumerate(chosen):
            filename = f"{dataset_name}_{attack_name}_eps_{eps:.2f}_{idx}.png".replace("/", "_")
            save_path = os.path.join(results_dir, filename)
            save_attack_triplet(
                x_img, x_adv_img, pred_o, pred_a, save_path, class_names=class_names
            )

    return success_rate, total