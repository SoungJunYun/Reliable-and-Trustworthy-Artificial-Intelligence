# Reliable and Trustworthy AI - Assignment #1

## Overview
This project implements adversarial attacks on neural networks for:
- MNIST
- CIFAR-10

Implemented attacks:
- Targeted FGSM
- Untargeted FGSM
- Targeted PGD
- Untargeted PGD

## Files
- `models.py`: neural network definitions
- `attack.py`: adversarial attack implementations
- `train.py`: training and evaluation
- `utils.py`: dataloaders, visualization, csv saving
- `test.py`: trains models and runs all attacks
- `results/`: saved PNG visualizations and CSV summary
- `requirements.txt`: Python dependencies
- `report.pdf`: short analysis report

## How to run
pip install -r requirements.txt
python test.py