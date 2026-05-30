# Fabric FL Prototype Distillation Demo

This repository contains a local simulation for federated learning with prototype
distillation. It is the algorithm-side prototype before replacing local payload
exchange with Fabric Private Data Collections.

## Files

- `examples/main.py`: experiment entry point and round orchestration.
- `examples/data.py`: dataset loading, IID/non-IID partitioning, and dataloaders.
- `examples/fl_client.py`: local client training, evaluation, prototype extraction, and update creation.
- `examples/models.py`: image classifier with a CNN feature extractor and embedding head.
- `examples/requirements.txt`: Python dependencies.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r examples/requirements.txt
```

## Fabric Binaries

Do not commit Fabric binaries into this repository. The local `bin/` directory is
ignored by Git because the binaries are large, platform-specific, and easy to
recreate.

For Hyperledger Fabric v3.1.4, use the official Fabric install script from the
Hyperledger Fabric repository:

```bash
curl -sSLO https://raw.githubusercontent.com/hyperledger/fabric/main/scripts/install-fabric.sh
chmod +x install-fabric.sh
./install-fabric.sh --fabric-version 3.1.4 binary
```

This downloads the platform-specific Fabric command binaries into `bin/`.

If you prefer manual download, the Fabric v3.1.4 release assets are available at:

- `https://github.com/hyperledger/fabric/releases/tag/v3.1.4`
- `https://github.com/hyperledger/fabric/releases/download/v3.1.4/hyperledger-fabric-linux-amd64-3.1.4.tar.gz`

## Run

MNIST:

```bash
python examples/main.py --dataset mnist
```

CIFAR-10:

```bash
python examples/main.py --dataset cifar10
```

CIFAR-100:

```bash
python examples/main.py --dataset cifar100
```

The script never downloads or extracts datasets automatically. Put the dataset
under `--data-dir` first. The default data directory is `data`.

## Datasets

MNIST official files:

- `https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz`
- `https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz`
- `https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz`
- `https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz`

Expected MNIST location:

```text
data/MNIST/raw/
```

CIFAR-10 official file:

- `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`

Expected CIFAR-10 location after extraction:

```text
data/cifar-10-batches-py/
```

CIFAR-100 official file:

- `https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz`

Expected CIFAR-100 location after extraction:

```text
data/cifar-100-python/
```

## Useful Quick Test

```bash
python examples/main.py \
  --dataset cifar10 \
  --num-clients 2 \
  --samples-per-client 40 \
  --rounds 1 \
  --local-epochs 1 \
  --batch-size 32 \
  --test-limit 200 \
  --compression int8
```

`--test-limit` is useful for CIFAR debugging because evaluating every client on
the full test set is slow on CPU.

## Key Options

- `--dataset`: `mnist`, `cifar10`, or `cifar100`.
- `--data-dir`: dataset cache directory. Defaults to `data`.
- `--num-clients`: number of simulated clients.
- `--samples-per-client`: local training samples per client.
- `--partition`: `iid` by default; use `noniid` for label-skewed clients.
- `--classes-per-client`: only used with `--partition noniid`.
- `--rounds`: number of federated rounds.
- `--local-epochs`: local epochs per round.
- `--proto-weight`: weight for global prototype distillation loss.
- `--compression`: prototype payload format: `fp32`, `fp16`, or `int8`.

## Current Flow

1. Load the selected dataset.
2. Split samples across logical clients.
3. Each client trains locally.
4. Each client computes class prototypes from its embedding space.
5. Prototypes and class counts are compressed into bytes.
6. The local aggregator decompresses and averages prototypes.
7. The next round uses global prototypes as a distillation target.

The payload bytes reported by the script are the stand-in for future Fabric PDC
payloads.
