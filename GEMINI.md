# GEMINI PROJECT INSTRUCTIONS — WS-Jev (Prediction Market Quantitative Research System)

## 1. Canonical Project Environment
* **Canonical Root Directory**: `/Users/admin/Documents/WS-Jev`.
* **Single Location Rule**: All future development, testing, CLI execution, datasets, reports, and Git operations MUST occur strictly inside `/Users/admin/Documents/WS-Jev`. Never execute development from `/Users/admin` or any temporary workspace.
* **Canonical Remote**: `https://github.com/Wandersport/WS-Jev.git`.

---

## 2. Non-Negotiable Safety & Structural Constraints
* **Strictly Paper-Only**: This system is exclusively for quantitative research, backtesting, probability calibration analysis, and paper trading simulation. It must remain **structurally incapable of live trading**.
* **Zero Live Execution Components**:
  * NO wallets, private keys, seed phrases, or keystore files.
  * NO blockchain transaction signing, broadcasting, or RPC execution calls.
  * NO authenticated trading API credentials (API keys, secrets, passphrases).
  * NO live execution adapters, live broker classes, placeholder broker stubs, or execution ABCs/protocols designed for future real trading substitution.
  * NO `--live`, `--real`, or live runtime switching switches.
  * NO `TODO` or `FIXME` comments planning or describing live trading execution.
* **Sole Permitted Execution Component**: `PaperBroker` local simulation engine.
* **Public Network Access**: Any external market data ingestion must be strictly public, unauthenticated, read-only GET requests with HTTPS schemes, explicit timeouts, strict host allowlists (`gamma-api.polymarket.com`, `clob.polymarket.com`), and bounded exponential backoff.

---

## 3. Quantitative & Engineering Principles
* **Priority Order**:
  1. Correctness
  2. Safety
  3. Reproducibility
  4. Transparency
  5. Probability Calibration
  6. Capital Preservation
  7. Statistical Validity
  8. Research Usefulness
  9. Simulated Returns
* **Zero Optimization Against Extreme Returns**: Do not optimize models or parameters toward unrealistic claims (e.g., turning small sums into thousands in hours). Focus on edge robustness, risk-adjusted survival, and calibration quality.
* **Timezone & Temporal Integrity**:
  * ALL timestamps must be timezone-aware and normalized to UTC (`ISO-8601`).
  * Never persist naive datetimes.
  * Strict look-ahead prevention: At simulated time $t$, algorithms and filters may only observe information available at or before $t$.
  * Never leak future resolution outcomes or settlement timestamps into active decision states.
* **Model Versioning & Auditability**:
  * Every probability estimate $\hat{q}$ and sizing decision must persist a clear `model_version`.
  * Every research cycle and replay run must log dataset IDs, configuration hashes, and random seeds.
* **Tooling & Dependencies**:
  * Use `uv` for virtual environments, package management, and tool execution (`uv run pytest`, `uv run ruff`, etc.).
  * Keep dependencies minimal and strictly justified (standard library preferred).

---

## 4. End-of-Session Workflow & Git Hygiene
Before concluding any autonomous session, if and only if all validation passes:
1. **Full Verification Suite**:
   ```bash
   uv sync
   uv run pytest
   uv run ruff check src tests
   uv run pmr verify-safety
   ```
2. **Public Safety Inspection**: Inspect `git status` and `git diff --cached` to guarantee:
   * Zero secrets, API keys, credentials, or personal tokens are tracked.
   * Zero machine-specific absolute path leaks or unrelated home-directory files are present.
   * Zero large transient datasets (`data/raw/`, `data/*.db`) or generated HTML reports are staged.
3. **Descriptive Local Commit**: Create clear, structured commits summarizing changes.
4. **Push to Remote**: Push to `origin` without force-pushing (`never force-push`).
5. **Verification**: Confirm remote push succeeded and report the resulting commit SHA.
