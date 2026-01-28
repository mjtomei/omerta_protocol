# Omerta Protocol

Protocol specifications, simulations, and academic papers for the Omerta project.

---

## Overview

This repository contains the formal protocol specifications for Omerta's distributed transaction system, along with simulation infrastructure for validating correctness and attack resistance.

---

## Protocol Specifications

Transaction specifications written in the Omerta transaction language (`.omt` files).

### Documentation

| Document | Description |
|----------|-------------|
| [DESIGN_PHILOSOPHY.md](docs/protocol/DESIGN_PHILOSOPHY.md) | Why lockless consensus, comparison to Lightning/blockchain/2PC |
| [FORMAT.md](docs/protocol/FORMAT.md) | Language specification for transaction state machines |
| [GENERATION.md](docs/protocol/GENERATION.md) | How schemas produce documentation and executable code |
| [GOSSIP.md](docs/protocol/GOSSIP.md) | Information propagation protocol |
| [SIMULATOR_DESIGN.md](docs/protocol/SIMULATOR_DESIGN.md) | Full simulation implementation details |

### Transaction Specifications

| Transaction | Status | Description |
|-------------|--------|-------------|
| [00_escrow_lock](docs/protocol/transactions/00_escrow_lock.md) | Tested | Lock funds with distributed witness consensus |
| [01_cabal_attestation](docs/protocol/transactions/01_cabal_attestation.md) | Tested | Witness attestation to compute session outcomes |
| [02_escrow_settle](docs/protocol/transactions/02_escrow_settle.md) | Draft | Release locked funds after service completion |
| [03_state_query](docs/protocol/transactions/03_state_query.md) | Draft | Query chain state from peers |
| [04_state_audit](docs/protocol/transactions/04_state_audit.md) | Draft | Third-party verification of chain consistency |
| [05_health_check](docs/protocol/transactions/05_health_check.md) | Draft | Liveness verification for witnesses |

Source `.omt` files are in [protocol/](protocol/).

---

## Simulations

Discrete event simulation infrastructure for validating protocol correctness and attack resistance.

- **Economic simulations**: Market dynamics, attack scenarios, monetary policy
- **Chain primitives**: Block structure, gossip protocol, network simulation
- **Transaction state machines**: Generated from `.omt` specifications
- **Framework**: Event-driven engine with AI agent integration

See [simulations/README.md](simulations/README.md) for details.

---

## Papers

The main technical papers (whitepaper and full participation verification paper) are in the top-level [omerta/papers/](../papers/) directory as PDFs with LaTeX sources.

This repository contains [simulation reports](papers/simulation-reports/) with empirical validation results.

## Plans

Working documents for development in [plans/](plans/):

| Category | Description |
|----------|-------------|
| [economic-analysis/](plans/economic-analysis/) | Market dynamics analysis |
| [mechanism-design/](plans/mechanism-design/) | Trust mathematics and defense mechanisms |
| [research/](plans/research/) | Analysis of related work |

---

## Development Status

See [plans/notes.txt](../plans/notes.txt) for the latest human-managed TODO list.

### Accomplished

- [x] Transaction 00 (escrow_lock) - tested and documented
- [x] Transaction 01 (cabal_attestation) - tested and documented
- [x] Discrete event simulation infrastructure
- [x] Economic simulation framework with market dynamics
- [x] Chain primitives (block structure, gossip protocol)
- [x] Event-driven simulation engine with AI agent integration
- [x] Design philosophy and format documentation

### TODO

- [ ] Complete Transaction 02 (escrow_settle) - currently draft
- [ ] Complete Transaction 03 (state_query) - currently draft
- [ ] Complete Transaction 04 (state_audit) - currently draft
- [ ] Complete Transaction 05 (health_check) - currently draft
- [ ] Run agent-based simulations to stress-test transactions 00 and 01
- [ ] Validate simulation results against real attack scenarios
- [ ] Protocol update mechanism with user consensus
- [ ] Machine fingerprinting to prevent fake verification responses
- [ ] Plan for intermittent remote backups of local chains

## Testing

```bash
# Install omerta_lang first
pip install -e ../omerta_lang

# Run simulation tests
pytest simulations/tests/
```

Tests automatically regenerate Python code from `.omt` files before running.
