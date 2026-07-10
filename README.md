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

The Go chaincode in `chaincode/` currently exposes `Set` and `Get`
transactions. Deployment is managed by
`fabric-network/scripts/deployChaincode.sh`.

## 3. Go Fabric Adapter

Start the persistent HTTP adapter:

```bash
./fabric-adapter/scripts/fabric-adapter.sh
curl http://127.0.0.1:8080/healthz
```

The adapter establishes one Fabric connection at startup and reuses it across
HTTP requests. The CLI remains available for diagnostics:

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
python fl/python/main.py --dataset mnist
python fl/python/main.py --dataset cifar10 --algorithm fedavg --rounds 30
python fl/python/main.py --dataset cifar10 --algorithm prototype --mode dirichlet --rounds 30
```

The Python training layer and Go Gateway client are currently runnable
independently. Their application-level integration is the next project step.
