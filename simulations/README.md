# Simulations

This directory contains simulation infrastructure for validating the Omerta protocol.

## Directory Structure

- `economic/` - Economic simulations (monetary policy, market dynamics, attack analysis)
- `chain/` - Chain primitives (blocks, transactions, gossip protocol)
- `transactions/` - Transaction state machine implementations
- `framework/` - Simulation framework (engine, runner, agents)
- `tests/` - Test suite

## Running Simulations

### Prerequisites

1. Install omerta_lang (the transaction language toolchain):
   ```bash
   pip install omerta_lang
   # Or for development:
   pip install -e ../omerta_lang
   ```

2. Install simulation dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### Running Tests

Tests automatically regenerate Python code from `.omt` files before running:

```bash
pytest simulations/tests/ -v
```

### Running Economic Simulations

```bash
# Individual simulations
python simulations/economic/trust_simulation.py
python simulations/economic/double_spend_simulation.py
python simulations/economic/monetary_policy_simulation.py

# Full study
python simulations/economic/run_full_study.py
```

## Simulation Modules

### Economic Simulations

| File | Description |
|------|-------------|
| `trust_simulation.py` | Models trust accumulation and decay |
| `double_spend_simulation.py` | Tests double-spend prevention |
| `monetary_policy_simulation.py` | Tests token emission and burn rates |
| `reliability_market_simulation.py` | Models compute reliability markets |
| `failure_modes.py` | Analyzes various failure scenarios |
| `identity_rotation_attack.py` | Tests Sybil resistance |

### Chain Primitives

| File | Description |
|------|-------------|
| `primitives.py` | Basic chain types (blocks, transactions) |
| `types.py` | Type definitions |
| `gossip.py` | Gossip protocol implementation |
| `network.py` | Network simulation |

### Transaction State Machines

| File | Description |
|------|-------------|
| `escrow_lock.py` | Escrow lock transaction |
| `cabal_attestation.py` | Cabal attestation transaction |
| `*_generated.py` | Auto-generated from `.omt` files |

### Framework

The simulation framework provides:
- Event-driven simulation engine
- Network topology simulation
- Agent-based modeling
- Trace recording and analysis
