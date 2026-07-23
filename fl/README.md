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

The default client data partition is Dirichlet beta. Select the paper-style
n-way k-shot partition with:

```bash
python fl/python/main.py \
  --dataset mnist \
  --algorithm prototype \
  --partition kn
```

K/N defaults to `ways=3`, `shots=100`, and `stdev=2`. Use
`python fl/python/main.py --help` to see all partition, attack, and training
parameters.

## Experiment Design

The experiments are organized by research question. Statistical heterogeneity,
model heterogeneity, Byzantine robustness, and Fabric overhead answer different
questions and should not be mixed into one accuracy table.

### Common Protocol

Use the following settings unless an experiment explicitly varies one of them:

| Setting | Value |
|---|---|
| Dataset | MNIST |
| Clients | 20, all participating in every round |
| Rounds | 100 |
| Local epochs | 1 |
| Batch size | 4 |
| Evaluation batch size | 256 |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Momentum | 0.5 |
| Prototype weight | 0.5 |
| Evaluation | Client-matched local and class-balanced global test sets |
| Test samples | 300 per client and 300 global |
| Seeds | 1234, 2024, 2025, 2026, 2027 |

Use one seed for smoke tests and at least three seeds for reported results.
Every method in a comparison must use the same partition seed, training budget,
test construction, and attack seed.

Beta experiments first select one seed-specific, class-balanced sample pool.
For MNIST with 20 clients and 300 samples per client, this pool contains 600
examples from each class. Changing beta only redistributes this same pool among
clients: every client still receives exactly 300 disjoint examples and the
global class histogram remains fixed. Each local test distribution matches its
client's training distribution. K/N experiments use 15 test samples per locally
present class.

The primary accuracy statistic is the mean local accuracy over the last 10
rounds. Global accuracy over the same rounds is a secondary generalization
metric. Report each metric's mean and standard deviation across seeds.
Final-round accuracy and best accuracy may be reported as secondary statistics,
but do not select a method using its best test round. Hyperparameters must be
selected before the final test runs or on a separate validation split.

### RQ1: Statistical Heterogeneity

This experiment asks whether prototype exchange improves independent local
training and how closely it approaches full-model aggregation when FedAvg is
applicable.

Run the following algorithms for each Dirichlet beta in `10.0`, `1.0`, `0.5`,
`0.2`, and `0.1`:

| Algorithm | Purpose |
|---|---|
| `local` | No-communication baseline |
| `fedavg` | Full-model aggregation baseline |
| `fedprox` | Non-IID full-model baseline |
| `prototype` | FedProto with the memory backend |

Command template:

```bash
python fl/python/main.py \
  --dataset mnist \
  --algorithm ALGORITHM \
  --partition beta \
  --beta BETA \
  --samples-per-client 300 \
  --num-clients 20 \
  --rounds 100 \
  --local-epochs 1 \
  --eval-scope both \
  --eval-batch-size 256 \
  --test-limit 300 \
  --seed SEED \
  --attack none
```

The main comparisons are:

```text
Prototype - Local       collaborative value of shared prototypes
FedAvg - Prototype      accuracy cost of communicating only prototypes
FedAvg / Prototype      logical communication reduction
```

Run the complete beta comparison automatically with:

```bash
python fl/scripts/run_beta_sweep.py \
  --dataset mnist \
  --betas 10.0 1.0 0.5 0.2 0.1 \
  --algorithms local fedavg fedprox prototype \
  --seeds 1234 \
  --rounds 10
```

This is a smoke-test matrix. For reported results, use the preregistered seeds
and full training budget:

```bash
python fl/scripts/run_beta_sweep.py \
  --dataset mnist \
  --betas 10.0 1.0 0.5 0.2 0.1 \
  --algorithms local fedavg fedprox prototype \
  --seeds 1234 2024 2025 2026 2027 \
  --rounds 100
```

Each invocation creates a timestamped directory under `fl/experiments/`. Its
`manifest.csv` records every command, status, log path, final/best accuracy, and
the mean accuracy over the last 10 rounds. `summary.csv` groups those results
by beta and algorithm and reports local/global accuracy plus their paired
differences from Local using the same beta and seed.

An interrupted or failed matrix can continue without rerunning completed jobs:

```bash
python fl/scripts/run_beta_sweep.py \
  --resume fl/experiments/beta_sweep_YYYY-MM-DD_HH-MM-SS
```

Use `--dry-run` to inspect all generated commands before training. The sweep
uses the memory backend, both evaluation scopes, a 256-example evaluation
batch, 300 local/global test samples, and no attacks so beta is the only
intended data variable.

### RQ2: Paper-Style K/N Partition

This experiment studies heterogeneous local label spaces using the paper-style
`3-way 100-shot` partition. It is not treated as an exact reproduction of the
published table because this project uses an independently verified evaluation
pipeline and sample-count-weighted prototype aggregation.

Run `local`, `fedavg`, `fedprox`, and `prototype` with:

```bash
python fl/python/main.py \
  --dataset mnist \
  --algorithm ALGORITHM \
  --partition kn \
  --ways 3 \
  --shots 100 \
  --stdev 2 \
  --train-shots-max 110 \
  --test-shots-per-class 15 \
  --num-clients 20 \
  --rounds 100 \
  --local-epochs 1 \
  --seed SEED \
  --attack none
```

For a paper-parameter reference run, use `--proto-weight 1.0`. Keep the
project's preregistered `--proto-weight 0.5` for the main comparison. Do not
stop Local early merely to match the accuracy reported by another paper; all
methods must receive the same local training budget.

### RQ3: Prototype Ablation

This experiment verifies that improvement comes from the global prototype
regularizer rather than from an unrelated training difference.

Run Prototype with `--proto-weight` in:

```text
0.0, 0.1, 0.5, 1.0
```

Use both `beta=0.5` and `beta=0.2`. With the same seed, Prototype at weight
`0.0` should closely match Local because exchanged prototypes do not affect
optimization. Weight `0.5` is the primary setting; the other values are an
ablation and must not be selected using the final test set.

### RQ4: Model Heterogeneity

This experiment asks whether prototype communication remains useful when model
parameters are structurally incompatible.

Planned client groups:

| Clients | CNN output channels | Prototype dimension |
|---|---:|---:|
| 0-6 | 18 | 50 |
| 7-13 | 20 | 50 |
| 14-19 | 22 | 50 |

The models have incompatible parameter shapes but share a 50-dimensional
prototype space. Compare:

| Method | Status in model-heterogeneous setting |
|---|---|
| Local | Valid no-communication baseline |
| FedProto memory | Valid |
| Fabric-FedProto | Valid |
| Vanilla FedAvg | N/A: parameter tensors are incompatible |
| FedMD/FedDF | Optional heterogeneous-model baseline requiring implementation |

The required result is not "FedProto beats FedAvg", because vanilla FedAvg
cannot execute in this setting. The meaningful claim is that FedProto improves
over Local while enabling collaboration across incompatible models.

Model-heterogeneous architectures are an experimental design target and are not
yet exposed by the current CLI. Do not report this experiment until the model
groups and a heterogeneous-compatible baseline are implemented and tested.

### RQ5: Attack Robustness

Use the same clean partition and seed as RQ1. Evaluate `sign_flip`, `noise`, and
`scale` attacks with malicious fractions `0.1`, `0.2`, and `0.3`.

Compare:

```text
fedavg
trimmed_mean
multi_krum
prototype --backend memory
prototype --backend fabric
```

Attack command template:

```bash
python fl/python/main.py \
  --dataset mnist \
  --algorithm ALGORITHM \
  --partition beta \
  --beta 0.5 \
  --samples-per-client 300 \
  --num-clients 20 \
  --rounds 100 \
  --local-epochs 1 \
  --malicious-fraction MALICIOUS_FRACTION \
  --attack ATTACK \
  --attack-scale 10 \
  --seed SEED \
  --attack-seed SEED
```

Trimmed Mean and Multi-Krum are primarily attack baselines. In a clean run,
Trimmed Mean has an automatically calculated trim count of zero and adds little
information beyond FedAvg.

Report:

```text
clean accuracy
attacked accuracy
accuracy drop from the paired clean run
detection precision, recall, F1, FPR, and false negatives
accuracy before and after malicious clients are excluded
```

Fabric uses two warm-up rounds before reputation-based exclusion. Report
detection metrics separately for warm-up and filtering rounds. The
`targeted_label_flip` attack is Prototype-specific and should be reported in a
separate table with its source-to-target success rate.

### RQ6: Fabric Cost and Correctness

Memory and Fabric backends implement the same Prototype experiment. Their
accuracy should be nearly identical in clean runs; Fabric is not expected to
increase accuracy by itself.

Start the network and adapter, then run:

```bash
python fl/python/main.py \
  --dataset mnist \
  --algorithm prototype \
  --backend fabric \
  --fabric-traffic \
  --partition beta \
  --beta 0.5 \
  --samples-per-client 300 \
  --num-clients 20 \
  --rounds 100 \
  --local-epochs 1 \
  --seed SEED \
  --attack none
```

Compare the paired memory and Fabric runs using:

| Metric | Interpretation |
|---|---|
| Local prototype payload | Algorithm-level communication |
| Fabric RX/TX traffic | Real peer and orderer container traffic |
| Accuracy difference | Fixed-point and backend correctness |
| Detection metrics | Security benefit under attack |
| Ledger round status | Auditability and deterministic finalization |

Run Fabric experiments without unrelated network workloads. Real Fabric traffic
includes endorsement, ordering, block propagation, and protocol overhead, so it
must not be presented as if it were only the prototype tensor size.

### Recommended Execution Order

1. Run five-round smoke tests for Local, FedAvg, and Prototype.
2. Complete RQ1 with one seed to validate the full pipeline.
3. Run the final RQ1 and RQ2 matrix with at least three seeds.
4. Complete the Prototype weight ablation.
5. Run memory attack baselines before starting Fabric experiments.
6. Run paired memory/Fabric clean and attack experiments.
7. Implement and validate model heterogeneity before reporting RQ4.

Keep all generated logs. Record the Git commit, command line, seed, dataset
files, chaincode version, and Fabric image versions used for every reported
result.

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
