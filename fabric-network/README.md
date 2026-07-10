# Fabric Network

This directory owns all Fabric infrastructure and generated network material.

```text
fabric-network/
├── config/             # Peer CLI configuration
├── configtx/           # Channel configuration
├── crypto-config/      # Cryptogen organization definitions
├── organizations/      # Generated MSP and TLS identities
├── channel-artifacts/  # Generated channel blocks
├── scripts/            # Create, join, and deploy operations
├── docker-compose.yaml
└── network.sh
```

Run commands from the repository root, for example:

```bash
./fabric-network/network.sh ps
```

The Compose project name is fixed to `fabric-fl` so moving this directory does
not create a second set of ledger volumes.
