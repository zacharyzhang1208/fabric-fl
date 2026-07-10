# Chaincode

The `contracts` Go chaincode is the ledger-facing contract for the project. It
currently provides:

- `Set(key, value)` to write state
- `Get(key)` to read state

Deploy it from the repository root:

```bash
./fabric-network/scripts/deployChaincode.sh
```
