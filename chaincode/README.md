# Chaincode

The `contracts` Go chaincode is the ledger-facing contract for the project. Its
prototype training flow provides:

- `CreateRound(roundID, expectedClients, numClasses, dimension, scale)`
- `SubmitPrototype(roundID, clientID, payloadJSON)`
- `FinalizeRound(roundID)`
- `GetGlobalPrototype(roundID)`

`FinalizeRound` validates that every expected client submitted a prototype,
performs deterministic fixed-point equal-client-weight aggregation for each
class, stores the global prototype, and marks the round as finalized in one
transaction.

Generic `Set/Get` transactions remain available for diagnostics, but `Set`
cannot write keys in the round, local prototype, or global prototype reserved
namespaces.

Deploy it from the repository root:

```bash
./fabric-network/scripts/deployChaincode.sh
```
