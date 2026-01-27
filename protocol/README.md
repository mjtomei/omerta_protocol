# Protocol Specifications

This directory contains the formal protocol specifications in the Omerta transaction language (`.omt` files).

## Directory Structure

- `shared/` - Shared type definitions used across transactions
- `transactions/` - Individual transaction specifications

## Transactions

| Directory | Transaction | Description |
|-----------|-------------|-------------|
| `00_escrow_lock/` | Escrow Lock | Locks funds in escrow for compute verification |
| `01_cabal_attestation/` | Cabal Attestation | Multi-party attestation of compute results |
| `02_escrow_settle/` | Escrow Settle | Releases escrowed funds after verification |
| `03_state_query/` | State Query | Queries chain state |
| `04_state_audit/` | State Audit | Audits state consistency |
| `05_health_check/` | Health Check | Node health verification |

## Working with `.omt` Files

### Linting

Check for errors and warnings:
```bash
omerta-lint protocol/transactions/00_escrow_lock/transaction.omt
omerta-lint --all  # Lint all transactions
```

### Code Generation

Generate Python state machines and documentation:
```bash
omerta-generate protocol/transactions/00_escrow_lock --python --output-dir simulations/transactions/
omerta-generate protocol/transactions/00_escrow_lock --markdown
```

### Regenerate All

Regenerate all transaction artifacts:
```bash
omerta-regenerate --protocol-dir protocol/ --python-output simulations/transactions/
```

## Language Reference

See `../docs/protocol/FORMAT.md` for the transaction language specification.
