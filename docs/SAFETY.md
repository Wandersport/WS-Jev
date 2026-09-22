# Non-Negotiable Paper-Only Safety Policy

**STRUCTURALLY INCAPABLE OF LIVE TRADING**

This repository is an academic and quantitative research system designed exclusively for backtesting, calibration analysis, and paper simulation. It is intentionally and architecturally incapable of executing real-money trades.

---

## 1. Zero Live Trading Capability

The codebase does not contain:
- **No Wallets**: No private key storage, mnemonic phrases, BIP-39 generators, keystore files, or hardware wallet connectors.
- **No Blockchain Signing**: No `eth_account`, `web3`, `ethers`, Solana, or cryptographic signing libraries.
- **No Transaction Broadcasting**: No JSON-RPC endpoints, broadcast nodes, smart contract calls, or gas estimators.
- **No Exchange Trading Clients**: No authenticated API clients, HMAC secret keys, exchange order submission endpoints, or execution SDKs.
- **No Execution Interfaces**: No abstract base classes (`BaseBroker`, `BrokerInterface`) or placeholder stubs created with the intent of future live substitution.
- **No Live Flags**: No `--live`, `--real`, `--production`, or runtime paper/live switching parameters.

---

## 2. Sole Execution Implementation: PaperBroker

The only execution component that exists in this repository is [`PaperBroker`](../src/pm_research/execution/paper_broker.py).

`PaperBroker` operates strictly within local memory and local SQLite storage:
- It receives an approved `RiskDecision` and `TradeProposal`.
- It calculates hypothetical execution against visible order-book depth levels or a conservative mathematical slippage formula.
- It records a hypothetical `PaperOrder` and `PaperFill` locally in SQLite.
- It never makes network calls or dispatches orders outside the local process.

---

## 3. Read-Only Public Market Data Constraints

When consuming public market data via [`PublicMarketDataAdapter`](../src/pm_research/data/public_adapter.py):
- Only public, unauthenticated HTTP endpoints are queried.
- No API keys, passwords, or trading credentials are required, accepted, or stored.
- Requests utilize strict timeouts (5.0s) and polite rate limiting.
- On any network failure or rate limit: the adapter yields an empty result set (`no data -> no proposal`).
- Missing values are never fabricated.

---

## 4. Automated Safety Verification

To prove compliance with these safety constraints, the system includes an automated AST and static code analyzer:

```bash
uv run pmr verify-safety
```

The verifier executes:
1. **Forbidden Pattern Scanner**: Scans all Python source files to ensure complete absence of private keys, wallet mnemonics, transaction broadcasting routines, and live execution classes.
2. **AST Execution Check**: Enforces that `PaperBroker` is the only class in `execution/` and verifies that no abstract base class inheritance exists.
3. **AST CLI Check**: Verifies that no live trading flags (`--live`, `--real`, `--mode=live`) exist in `cli.py`.

Automated unit tests in `tests/test_safety.py` guarantee that safety checks are executed on every test run.
