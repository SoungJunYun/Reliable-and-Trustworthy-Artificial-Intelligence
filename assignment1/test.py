import os
import torch

from models import SimpleMNISTCNN, build_cifar10_model
from attack import (
    fgsm_targeted,
    fgsm_untargeted,
    pgd_targeted,
    pgd_untargeted,
)
from train import set_seed, fit_model, evaluate
from utils import (
    ensure_dir,
    get_mnist_loaders,
    get_cifar10_loaders,
    evaluate_attack,
    save_csv,
)


RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"

MNIST_EPOCHS = 4
CIFAR_EPOCHS = 2

MNIST_BATCH_SIZE = 128
CIFAR_BATCH_SIZE = 64

EVAL_SAMPLES = 100
EPS_VALUES = [0.05, 0.10, 0.20, 0.30]

# 시각화 저장은 중간 epsilon 하나에서만 수행
VIS_EPS = 0.10


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_dir(RESULTS_DIR)
    ensure_dir(CHECKPOINT_DIR)

    print(f"Using device: {device}")

    result_rows = []

    # ----------------------------
    # 1. MNIST
    # ----------------------------
    print("\n" + "=" * 70)
    print("Preparing MNIST...")
    mnist_train_loader, mnist_test_loader = get_mnist_loaders(
        batch_size=MNIST_BATCH_SIZE, data_root="./data"
    )

    mnist_model = SimpleMNISTCNN().to(device)
    mnist_model, best_mnist_acc = fit_model(
        mnist_model,
        mnist_train_loader,
        mnist_test_loader,
        device=device,
        epochs=MNIST_EPOCHS,
        lr=1e-3,
        save_path=os.path.join(CHECKPOINT_DIR, "mnist_cnn.pth"),
    )

    _, mnist_clean_acc = evaluate(mnist_model, mnist_test_loader, device)
    print(f"[MNIST] Clean Accuracy: {mnist_clean_acc:.2f}%")

    mnist_attacks = [
        ("fgsm_targeted", fgsm_targeted, True, {}),
        ("fgsm_untargeted", fgsm_untargeted, False, {}),
        ("pgd_targeted", pgd_targeted, True, {"k": 10}),
        ("pgd_untargeted", pgd_untargeted, False, {"k": 10}),
    ]

    for attack_name, attack_fn, targeted, base_kwargs in mnist_attacks:
        for eps in EPS_VALUES:
            attack_kwargs = dict(base_kwargs)
            if "pgd" in attack_name:
                attack_kwargs["eps_step"] = eps / 10.0

            success_rate, total = evaluate_attack(
                model=mnist_model,
                loader=mnist_test_loader,
                attack_fn=attack_fn,
                dataset_name="mnist",
                attack_name=attack_name,
                eps=eps,
                device=device,
                results_dir=RESULTS_DIR,
                targeted=targeted,
                attack_kwargs=attack_kwargs,
                class_names=None,
                num_classes=10,
                num_samples=EVAL_SAMPLES,
                save_examples=(eps == VIS_EPS),
            )

            print(
                f"[MNIST][{attack_name}][eps={eps:.2f}] "
                f"Success Rate: {success_rate:.2f}% ({total} samples)"
            )

            result_rows.append({
                "dataset": "MNIST",
                "attack": attack_name,
                "eps": eps,
                "success_rate": round(success_rate, 4),
                "clean_accuracy": round(mnist_clean_acc, 4),
                "num_eval_samples": total,
            })

    # ----------------------------
    # 2. CIFAR-10
    # ----------------------------
    print("\n" + "=" * 70)
    print("Preparing CIFAR-10...")
    cifar_train_loader, cifar_test_loader, cifar_classes = get_cifar10_loaders(
        batch_size=CIFAR_BATCH_SIZE,
        data_root="./data",
        image_size=224,  # pretrained resnet18 활용을 위해 224 사용
    )

    cifar_model = build_cifar10_model(
        use_pretrained=True,
        freeze_backbone=False
    ).to(device)

    cifar_model, best_cifar_acc = fit_model(
        cifar_model,
        cifar_train_loader,
        cifar_test_loader,
        device=device,
        epochs=CIFAR_EPOCHS,
        lr=1e-4,
        save_path=os.path.join(CHECKPOINT_DIR, "cifar10_resnet18.pth"),
    )

    _, cifar_clean_acc = evaluate(cifar_model, cifar_test_loader, device)
    print(f"[CIFAR-10] Clean Accuracy: {cifar_clean_acc:.2f}%")

    cifar_attacks = [
        ("fgsm_targeted", fgsm_targeted, True, {}),
        ("fgsm_untargeted", fgsm_untargeted, False, {}),
        ("pgd_targeted", pgd_targeted, True, {"k": 10}),
        ("pgd_untargeted", pgd_untargeted, False, {"k": 10}),
    ]

    for attack_name, attack_fn, targeted, base_kwargs in cifar_attacks:
        for eps in EPS_VALUES:
            attack_kwargs = dict(base_kwargs)
            if "pgd" in attack_name:
                attack_kwargs["eps_step"] = eps / 10.0

            success_rate, total = evaluate_attack(
                model=cifar_model,
                loader=cifar_test_loader,
                attack_fn=attack_fn,
                dataset_name="cifar10",
                attack_name=attack_name,
                eps=eps,
                device=device,
                results_dir=RESULTS_DIR,
                targeted=targeted,
                attack_kwargs=attack_kwargs,
                class_names=cifar_classes,
                num_classes=10,
                num_samples=EVAL_SAMPLES,
                save_examples=(eps == VIS_EPS),
            )

            print(
                f"[CIFAR-10][{attack_name}][eps={eps:.2f}] "
                f"Success Rate: {success_rate:.2f}% ({total} samples)"
            )

            result_rows.append({
                "dataset": "CIFAR-10",
                "attack": attack_name,
                "eps": eps,
                "success_rate": round(success_rate, 4),
                "clean_accuracy": round(cifar_clean_acc, 4),
                "num_eval_samples": total,
            })

    csv_path = os.path.join(RESULTS_DIR, "attack_success_rates.csv")
    save_csv(result_rows, csv_path)
    print(f"\nSaved summary CSV to: {csv_path}")


if __name__ == "__main__":
    main()