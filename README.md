# Fabric FL

Hyperledger Fabric-backed federated learning project. The repository is split
into four layers with independent responsibilities.

```text
fabric-fl/
├── fabric-network/   # Fabric network, identities, channel, and operations
├── chaincode/        # Smart contract installed on the Fabric network
├── fabric-adapter/   # Go adapter for accessing Fabric
└── fl/               # Python federated learning code, datasets, and logs
```

## Local Quick Start

Run everything from the repository root.

Make sure the Fabric network is running:

```bash
./fabric-network/network.sh up
./fabric-network/network.sh ps
```

After a fresh network creation or `network.sh reset`, join the channel and
deploy the chaincode once before starting the adapter:

```bash
./fabric-network/scripts/joinChannel.sh
./fabric-network/scripts/deployChaincode.sh
```

Start the persistent Go adapter in the background:

```bash
./fabric-adapter/scripts/fabric-adapter.sh start
```

Activate Python and optionally check the adapter:

```bash
source .venv/bin/activate
curl http://127.0.0.1:18080/healthz
```

Run a small two-client, two-round MNIST experiment whose prototypes are
aggregated by the Fabric chaincode:

```bash
python fl/python/main.py \
  --dataset mnist \
  --algorithm prototype \
  --backend fabric \
  --num-clients 2 \
  --rounds 2 \
  --local-epochs 1 \
  --samples-per-client 30 \
  --beta 0.5 \
  --batch-size 4 \
  --test-limit 20
```

The MNIST files must exist under `fl/data/MNIST/raw/`. Training output is
printed in the terminal and saved under `fl/log/`. Each run automatically selects
new ledger round IDs and a new reputation experiment ID, so independent runs do
not share client scores.

To exercise reputation filtering, use more than two rounds and enough clients
for at least three contributors per class. For example:

```bash
python fl/python/main.py \
  --dataset mnist \
  --algorithm prototype \
  --backend fabric \
  --num-clients 20 \
  --rounds 4 \
  --local-epochs 1 \
  --malicious-fraction 0.15 \
  --attack sign_flip \
  --attack-scale 10
```

The training log shows anomalous and excluded client IDs, on-chain reputation
scores, and experiment-only precision/recall/F1/FPR metrics.

After the experiment, stop the adapter and then stop the Fabric containers
without deleting ledger volumes when they are no longer needed:

```bash
./fabric-adapter/scripts/fabric-adapter.sh stop
```

```bash
./fabric-network/network.sh down
```

## 1. Fabric Network

Network commands are run from the repository root:

```bash
./fabric-network/network.sh up
./fabric-network/network.sh ps
./fabric-network/network.sh down
```

To generate a new network and deploy the chaincode:

```bash
./fabric-network/scripts/createOrgs.sh
./fabric-network/scripts/generateChannelArtifacts.sh
./fabric-network/network.sh up
./fabric-network/scripts/joinChannel.sh
./fabric-network/scripts/deployChaincode.sh
```

`network.sh reset` removes the Fabric ledger volumes. Use it only when a full
network reset is intended.

## 2. Chaincode

The Go chaincode in `chaincode/` processes each complete prototype round in one
atomic transaction, including deterministic fixed-point aggregation, reputation
updates, and a canonical input-batch hash. Raw prototypes remain in the
immutable transaction input instead of being duplicated in world state.
Generic `Set/Get` and staged prototype
transactions remain available for diagnostics. Deployment is managed by
`fabric-network/scripts/deployChaincode.sh`.

## 3. Go Fabric Adapter

Start the persistent HTTP adapter:

```bash
./fabric-adapter/scripts/fabric-adapter.sh start
```

Check its health:

```bash
curl http://127.0.0.1:18080/healthz
```

The adapter establishes one Fabric connection at startup and reuses it across
HTTP requests. For prototype training it buffers separate logical-client
submissions and forwards the complete round using one Fabric batch upload
transaction. After the atomic write commits, the global prototype and
reputation report are retrieved through two single-peer queries. The CLI remains
available for diagnostics:

After the network is running and the `contracts` chaincode is deployed:

```bash
./fabric-adapter/scripts/fabric-cli.sh set hello world
./fabric-adapter/scripts/fabric-cli.sh get hello
```

See `fabric-adapter/README.md` for configuration and generic transaction calls.

## 4. Python Federated Learning

Create the Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r fl/python/requirements.txt
```

Datasets are stored under `fl/data/`; training logs are written to `fl/log/`.
Both defaults are resolved from the script location, so the training command
works from any current directory:

```bash
python fl/python/main.py --dataset mnist --algorithm prototype
python fl/python/main.py --dataset cifar10 --algorithm fedavg --rounds 30
python fl/python/main.py --dataset cifar10 --algorithm prototype --beta 0.5 --rounds 30
```

Run prototype aggregation through the Fabric chaincode after starting the HTTP
adapter in a separate terminal:

```bash
python fl/python/main.py --dataset mnist --algorithm prototype --backend fabric
```

The default prototype backend remains `memory` so local baseline experiments do
not require a running Fabric network.

The Python training loop can upload typed prototype rounds, receive finalized
global prototypes and reputation reports through the HTTP adapter,
and use the retrieved result in the next training round.

Training logs report bidirectional raw-tensor communication as
`logical_communication` (`upload`, `download`, and `round_total`). Fabric runs
add a separate `fabric_traffic` metric for aggregate peer/orderer container
RX+TX, including consensus and protocol overhead.

See `fl/README.md` for the statistical heterogeneity, K/N, attack robustness,
model heterogeneity, and Fabric overhead experiment protocol.
