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
python fl/python/main.py --dataset mnist --algorithm prototype
```

Use the Fabric chaincode as the prototype aggregation backend:

Start the adapter in terminal 1:

```bash
./fabric-adapter/scripts/fabric-adapter.sh
```

The default adapter URL is `http://127.0.0.1:18080`. Set
`FABRIC_ADAPTER_URL` or pass `--fabric-adapter-url` when using a different
address.

Run training in terminal 2:

```bash
source .venv/bin/activate
python fl/python/main.py \
  --dataset mnist \
  --algorithm prototype \
  --backend fabric
```

Each run automatically chooses a millisecond-based ledger round base. Set
`--fabric-round-base` explicitly when a reproducible ledger identifier is
needed. A previously finalized ledger round cannot accept new submissions.

Use `python fl/python/main.py --help` to see the available algorithms,
beta partition, attack settings, and training parameters.

## Prototype HTTP Transport

`python/fabric_adapter.py` defines the prototype wire format and HTTP client.
It quantizes float tensors to deterministic fixed-point `int64` values before
sending them to the persistent Fabric Adapter.

```python
from fabric_adapter import FabricAdapterClient, PrototypePayload

payload = PrototypePayload.from_tensors(
    round_id=1,
    client_id=client.client_id,
    prototypes=update.prototypes,
    counts=update.counts,
)

adapter = FabricAdapterClient()
adapter.create_round(1, 1, 1, len(clients), 10, 50)
adapter.upload_prototype(payload)
adapter.finalize_round(1)
global_payload = adapter.get_global_prototype(1)
global_prototypes, global_counts = global_payload.to_tensors(device="cpu")
```

Prototype writes use the dedicated `SubmitPrototype` chaincode transaction.
`FinalizeRound` evaluates each logical client ID, updates its experiment-scoped
reputation, excludes repeatedly anomalous clients after two warm-up rounds,
and stores the filtered global prototype. The round reputation report contains
the on-chain distances, decisions, scores, and statuses. Simulation attack
labels are used only to print detection metrics and are not sent to Fabric.
