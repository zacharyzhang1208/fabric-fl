# Chaincode

The `contracts` Go chaincode is the ledger-facing contract for the project. Its
prototype training flow provides:

- `CreateRound(roundID, experimentID, sequence, expectedClients, numClasses, dimension, scale)`
- `SubmitPrototype(roundID, clientID, payloadJSON)`
- `FinalizeRound(roundID)`
- `GetGlobalPrototype(roundID)`
- `GetRoundReputationReport(roundID)`
- `GetClientReputation(experimentID, clientID)`

`FinalizeRound` validates every expected prototype, scores each logical client
ID with a deterministic median/MAD detector, updates its experiment-scoped
reputation, filters repeatedly anomalous clients, and stores both the global
prototype and an auditable report in one transaction. The first two sequences
are warm-up rounds: assessments and scores are recorded, but nobody is
excluded.

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
