# Fabric FL Simulation

Federated learning simulation for Hyperledger Fabric. The current Python demo is
under `examples/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r examples/requirements.txt
```

## Data

Datasets are not downloaded or extracted by the program. Put them under `data/`
yourself:

```text
data/MNIST/raw/
data/cifar-10-batches-py/
data/cifar-100-python/
```

Then run, for example:

```bash
python examples/main.py --dataset mnist
python examples/main.py --dataset cifar10
python examples/main.py --dataset cifar100
```

Compare prototype aggregation with standard FedAvg on the same split:

```bash
python examples/main.py --dataset cifar10 --mode prototype --partition noniid --classes-per-client 2 --rounds 30
python examples/main.py --dataset cifar10 --mode fedavg --partition noniid --classes-per-client 2 --rounds 30
```
