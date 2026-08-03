# Chaincode

The `contracts` Go chaincode is the ledger-facing contract for the project. Its
prototype training flow provides:

- `ProcessRound(roundID, experimentID, sequence, expectedClients, numClasses, dimension, scale, payloadsJSON)`
- `CreateRound(roundID, experimentID, sequence, expectedClients, numClasses, dimension, scale)`
- `SubmitPrototype(roundID, clientID, payloadJSON)`
- `SubmitPrototypeBatch(roundID, payloadsJSON)`
- `FinalizeRound(roundID)`
- `GetGlobalPrototype(roundID)`
- `GetRoundReputationReport(roundID)`
- `GetClientReputation(experimentID, clientID)`

`ProcessRound` is the normal training path. It validates every expected
prototype, scores each logical client ID with a deterministic median/MAD
detector, updates its experiment-scoped
reputation, filters repeatedly anomalous clients, and stores both the global
prototype and an auditable report atomically. Its transaction response contains
only the finalized round ID and status; Python queries the global prototype and
report from one peer after commit. The first two sequences are warm-up rounds:
assessments and scores are recorded, but nobody is excluded.

The Adapter collects one submission from each logical client, orders the
complete batch by client ID, and invokes `ProcessRound` once. The chaincode
rejects incomplete, duplicate, conflicting, or malformed batches. The complete
prototype batch remains in the immutable Fabric transaction input, while world
state stores its canonical SHA-256 instead of duplicating 20 prototype records.
Individual assessments are stored once inside the round report rather than as
separate state entries. An identical complete batch may be retried safely after
an uncertain network response; a different hash is rejected. The original
three-stage transactions retain their per-client state records for diagnostics.

The atomic path therefore keeps current query state limited to the round,
global prototype, complete report, experiment sequence, and current client
reputations. Recovering a historical client's raw prototype requires reading and
decoding the original `ProcessRound` transaction rather than a world-state key.

Within one `experimentID`, sequences must be created in order. The previous
sequence must already be finalized, and the client count, prototype shape, and
fixed-point scale cannot change during the experiment.

Scores range from 0 to 10000 and start at 8000. Each assessment uses an 80/20
moving average of the old reputation and current binary score. Scores at or
above 7000 are `TRUSTED`, 5000 through 6999 are `WATCH`, and lower scores are
`BLOCKED`. From sequence 3 onward, two consecutive anomalies cause exclusion.
If filtering leaves a class empty, its coordinate-wise median is used as a
robust fallback.

Participants are currently identified by submitted `clientID`; IDs are not yet
bound to unique Fabric certificates. Python attack labels are used only for
experiment metrics and are never sent to chaincode.

Generic `Set/Get` transactions remain available for diagnostics, but `Set`
cannot write keys in the round, local prototype, or global prototype reserved
namespaces.

Deploy it from the repository root:

```bash
./fabric-network/scripts/deployChaincode.sh
```
