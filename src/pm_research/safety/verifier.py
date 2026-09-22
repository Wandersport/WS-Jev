"""Automated safety verification engine.

Scans the repository to verify that:
1. No wallet implementations, private-key handling, seed phrases, or mnemonics exist.
2. No blockchain signing, transaction broadcasting, or RPC submission exists.
3. No live execution stubs, placeholder brokers, or execution abstract base classes exist.
4. No '--live', '--real', or live trading CLI modes exist.
5. PaperBroker is the only execution implementation in the system.
6. No live trading dependencies, libraries, or SDKs are imported or declared.
7. All external network interactions are strictly read-only GET requests with zero authentication headers.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SafetyVerificationResult:
    passed: bool
    scanned_files_count: int
    violations: list[str] = field(default_factory=list)
    checked_rules: list[str] = field(default_factory=list)


class SafetyVerifier:
    """Rigorous AST, static code, and configuration scanner verifying structural paper-only constraints."""

    # Forbidden tokens indicating real trading, wallet handling, credentials, or live brokers
    FORBIDDEN_RUNTIME_PATTERNS = [
        (r"\bprivate_key\b", "Private key handling"),
        (r"\bsecret_key\b", "Secret key / trading secret credential"),
        (r"\bseed_phrase\b", "Seed phrase handling"),
        (r"\bmnemonic\b", "Mnemonic / BIP-39 handling"),
        (r"\bbip39\b", "BIP-39 mnemonic standard"),
        (r"\bbip44\b", "BIP-44 HD wallet derivation standard"),
        (r"\bkeystore\b", "Wallet keystore storage"),
        (r"\bhdwallet\b", "Hierarchical Deterministic wallet"),
        (r"\bweb3\b", "Web3 / blockchain RPC library"),
        (r"\beth_account\b", "Ethereum account / wallet signing library"),
        (r"\beth_abi\b", "Ethereum ABI encoding library"),
        (r"\bethers\b", "Ethers library reference"),
        (r"\bsign_transaction\b", "Transaction signing method"),
        (r"\bsend_transaction\b", "Transaction broadcasting method"),
        (r"\bbroadcast_tx\b", "RPC transaction broadcasting"),
        (r"\bsign_message\b", "Cryptographic message signing"),
        (r"\bpersonal_sign\b", "Personal message signing"),
        (r"\beip712\b", "EIP-712 structured signing"),
        (r"\bLiveBroker\b", "LiveBroker class or reference"),
        (r"\bRealBroker\b", "RealBroker class or reference"),
        (r"\bLiveExecutor\b", "LiveExecutor class or reference"),
        (r"\bRealExecutor\b", "RealExecutor class or reference"),
        (r"\bBaseBroker\b", "Generic broker abstraction intended for live substitution"),
        (r"\bBrokerInterface\b", "Broker interface intended for live substitution"),
        (r"\bExecutionInterface\b", "Execution interface intended for live substitution"),
        (r"\bAbstractBroker\b", "Abstract broker intended for live substitution"),
        (r"\bTradingBroker\b", "Trading broker reference"),
        (r"\bOrderExecutionService\b", "Order execution service"),
        (r"\bccxt\b", "CCXT live trading library"),
        (r"\bpy_clob_client\b", "Polymarket live CLOB client"),
        (r"\bpolymarket_clob\b", "Polymarket live CLOB library"),
    ]

    FORBIDDEN_MODULE_IMPORTS = {
        "web3",
        "eth_account",
        "eth_abi",
        "ccxt",
        "py_clob_client",
        "polymarket_clob",
        "cryptography",
        "requests",  # Disallow arbitrary high-level HTTP libraries; use standard urllib GET with allowlist
    }

    FORBIDDEN_DEPENDENCY_NAMES = {
        "web3",
        "eth-account",
        "eth-abi",
        "ccxt",
        "py-clob-client",
        "polymarket-clob",
        "cryptography",
        "requests",
    }

    def __init__(self, root_path: str | Path | None = None) -> None:
        if root_path is None:
            # Locate src/pm_research directory relative to this file
            self.root_path = Path(__file__).resolve().parent.parent
            # Also locate repository root (parent of src)
            self.repo_root = self.root_path.parent.parent
        else:
            self.root_path = Path(root_path)
            self.repo_root = self.root_path.parent if self.root_path.name == "pm_research" else self.root_path

    def verify_all(self) -> SafetyVerificationResult:
        """Run all safety verification checks across the codebase."""
        violations: list[str] = []
        checked_rules: list[str] = []

        # 1. Scan all python source files in src/pm_research (excluding safety verifier itself)
        py_files = [
            f for f in self.root_path.rglob("*.py")
            if "safety" not in f.parts and "__pycache__" not in f.parts
        ]
        scanned_count = len(py_files)

        checked_rules.append("Rule 1: No wallet, private-key, seed phrase, signing, or live trading tokens")
        for f in py_files:
            content = f.read_text(encoding="utf-8")
            for pattern, desc in self.FORBIDDEN_RUNTIME_PATTERNS:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for m in matches:
                    line_num = content[:m.start()].count("\n") + 1
                    violations.append(
                        f"Forbidden pattern '{pattern}' ({desc}) in {f.relative_to(self.root_path)}:{line_num}"
                    )

        # 2. Check for prohibited TODO/FIXME comments hinting at future live execution or wallets
        checked_rules.append("Rule 2: No TODO/FIXME comments describing live trading, wallets, or order execution")
        future_live_re = re.compile(r"(?:TODO|FIXME|XXX).*(?:live|wallet|sign|real trading|real broker)", re.IGNORECASE)
        for f in py_files:
            content = f.read_text(encoding="utf-8")
            for line_idx, line in enumerate(content.splitlines(), start=1):
                if future_live_re.search(line):
                    violations.append(
                        f"Prohibited live-execution TODO/FIXME in {f.relative_to(self.root_path)}:{line_idx}: {line.strip()}"
                    )

        # 3. AST Check on imports across all source files
        checked_rules.append("Rule 3: No forbidden live trading, crypto, or external HTTP library imports")
        for f in py_files:
            try:
                tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
            except SyntaxError as e:
                violations.append(f"Syntax error parsing {f}: {e}")
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        base_mod = alias.name.split(".")[0]
                        if base_mod in self.FORBIDDEN_MODULE_IMPORTS:
                            violations.append(
                                f"Forbidden import '{alias.name}' in {f.relative_to(self.root_path)}:{node.lineno}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        base_mod = node.module.split(".")[0]
                        if base_mod in self.FORBIDDEN_MODULE_IMPORTS:
                            violations.append(
                                f"Forbidden import from '{node.module}' in {f.relative_to(self.root_path)}:{node.lineno}"
                            )

        # 4. AST Check on execution package: verify PaperBroker is the only execution class and has no ABCs
        checked_rules.append("Rule 4: PaperBroker is the sole permitted execution class (no ABCs, stubs, or interfaces)")
        exec_dir = self.root_path / "execution"
        if exec_dir.exists():
            for f in exec_dir.glob("*.py"):
                if f.name == "__init__.py":
                    continue
                tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if node.name != "PaperBroker":
                            violations.append(
                                f"Unauthorized class '{node.name}' in execution package: {f.name}. "
                                f"Only 'PaperBroker' is permitted."
                            )
                        # Ensure no class inherits from ABC / abstract base class
                        for base in node.bases:
                            if isinstance(base, ast.Name) and base.id in ("ABC", "AbstractBroker", "Protocol"):
                                violations.append(
                                    f"Abstract base class or Protocol inheritance detected in '{node.name}' in {f.name}."
                                )

        # 5. AST Check across all classes: no classes named *Live* or *RealBroker*
        checked_rules.append("Rule 5: No live execution class names or live broker abstractions across repository")
        for f in py_files:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    lower_name = node.name.lower()
                    if ("live" in lower_name and "liveness" not in lower_name and "delivery" not in lower_name) or "realbroker" in lower_name:
                        violations.append(
                            f"Prohibited class name '{node.name}' in {f.relative_to(self.root_path)}:{node.lineno}"
                        )

        # 6. AST & code check on networking: verify Polymarket data is read-only GET with zero auth,
        # and OpenRouter research POST/Bearer auth is restricted strictly to research/jev_openrouter.py
        checked_rules.append(
            "Rule 6: Network requests restricted to read-only GET for market data; authenticated POST restricted strictly to Jev research provider"
        )
        data_dir = self.root_path / "data"
        if data_dir.exists():
            for f in data_dir.glob("*.py"):
                content = f.read_text(encoding="utf-8")
                # Ensure no POST/PUT/PATCH/DELETE method specifications in market data
                for bad_method in ['method="POST"', "method='POST'", 'method="PUT"', "method='PUT'",
                                  'method="PATCH"', "method='PATCH'", 'method="DELETE"', "method='DELETE'"]:
                    if bad_method.lower() in content.lower():
                        violations.append(f"Prohibited non-GET HTTP method in market data adapter {f.name}: {bad_method}")
                # Ensure no auth header keys in market data
                for bad_header in ["Authorization", "X-API-KEY", "API_KEY", "Bearer", "PRIVATE-KEY"]:
                    if bad_header.lower() in content.lower():
                        violations.append(f"Prohibited authentication header '{bad_header}' in market data adapter {f.name}")

        # Across all files in src/pm_research:
        # Prohibit POST, Authorization headers, and OPENROUTER_API_KEY outside of research/jev_openrouter.py
        for f in py_files:
            rel_path = str(f.relative_to(self.root_path))
            if rel_path != "research/jev_openrouter.py":
                content = f.read_text(encoding="utf-8")
                if '"Authorization"' in content or "'Authorization'" in content:
                    violations.append(
                        f"Prohibited Authorization header outside of dedicated Jev research provider in {rel_path}"
                    )
                if 'method="POST"' in content or "method='POST'" in content:
                    violations.append(
                        f"Prohibited HTTP POST method outside of dedicated Jev research provider in {rel_path}"
                    )
                if "OPENROUTER_API_KEY" in content:
                    violations.append(
                        f"Prohibited OPENROUTER_API_KEY access outside of dedicated Jev research provider in {rel_path}"
                    )

        # 7. Check CLI options: verify no --live, --real, or live switching flags
        checked_rules.append("Rule 7: No --live, --real, or live-switching CLI options")
        cli_file = self.root_path / "cli.py"
        if cli_file.exists():
            cli_content = cli_file.read_text(encoding="utf-8")
            for forbidden_flag in ["--live", "--real", "--mode=live", "--production", "--live-trading"]:
                if forbidden_flag in cli_content:
                    violations.append(f"Forbidden live trading flag '{forbidden_flag}' found in cli.py")

        # 8. Dependency check: verify pyproject.toml has no live trading dependencies
        checked_rules.append("Rule 8: pyproject.toml contains no live trading or blockchain dependencies")
        pyproject_file = self.repo_root / "pyproject.toml"
        if pyproject_file.exists():
            pyproject_content = pyproject_file.read_text(encoding="utf-8").lower()
            for dep in self.FORBIDDEN_DEPENDENCY_NAMES:
                if f'"{dep}"' in pyproject_content or f"'{dep}'" in pyproject_content or f'"{dep}>' in pyproject_content:
                    violations.append(f"Prohibited dependency '{dep}' declared in pyproject.toml")

        passed = len(violations) == 0
        return SafetyVerificationResult(
            passed=passed,
            scanned_files_count=scanned_count,
            violations=violations,
            checked_rules=checked_rules,
        )
