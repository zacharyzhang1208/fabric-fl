# Federated Learning

```text
fl/
├── python/  # Training entry point, models, clients, and algorithms
├── data/    # Local MNIST and CIFAR datasets
└── log/     # Training output
```

Run from the repository root:

```bash
source .venv/bin/activate
python fl/python/main.py --dataset mnist
```

Use `python fl/python/main.py --help` to see the available algorithms,
partition modes, attack settings, and training parameters.
