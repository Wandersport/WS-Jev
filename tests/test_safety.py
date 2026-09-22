"""Automated safety tests verifying paper-trading only enforcement and zero live trading capability."""

from __future__ import annotations

from pm_research.safety.verifier import SafetyVerifier


def test_safety_verifier_passes_on_repo():
    """Verify that automated safety checks pass completely across the repository."""
    verifier = SafetyVerifier()
    result = verifier.verify_all()

    assert result.scanned_files_count > 0, "Safety verifier must scan active repository files."
    assert result.passed, f"Safety verification failed with violations: {result.violations}"
    assert len(result.violations) == 0


def test_safety_verifier_detects_prohibited_patterns(tmp_path):
    """Verify that the safety verifier positively catches any attempts to inject live execution code or wallets."""
    dummy_repo = tmp_path / "mock_src"
    dummy_repo.mkdir()
    (dummy_repo / "execution").mkdir()

    # Create dummy malicious file containing private key and live broker class
    bad_code = """
class LiveBroker:
    def __init__(self, private_key: str):
        self.private_key = private_key
    def sign_transaction(self, tx):
        pass
"""
    (dummy_repo / "execution" / "live_broker.py").write_text(bad_code, encoding="utf-8")

    verifier = SafetyVerifier(root_path=dummy_repo)
    result = verifier.verify_all()

    assert not result.passed
    assert any("private_key" in v for v in result.violations)
    assert any("LiveBroker" in v for v in result.violations)


def test_safety_verifier_detects_forbidden_imports(tmp_path):
    """Verify safety verifier detects attempts to import web3, ccxt, or live trading SDKs."""
    dummy_repo = tmp_path / "mock_src"
    dummy_repo.mkdir()
    (dummy_repo / "data").mkdir()

    bad_code = """
import ccxt
from web3 import Web3

def get_data():
    pass
"""
    (dummy_repo / "data" / "bad_feed.py").write_text(bad_code, encoding="utf-8")

    verifier = SafetyVerifier(root_path=dummy_repo)
    result = verifier.verify_all()

    assert not result.passed
    assert any("ccxt" in v for v in result.violations)
    assert any("web3" in v for v in result.violations)


def test_safety_verifier_detects_prohibited_http_methods(tmp_path):
    """Verify safety verifier detects POST/PUT/DELETE method calls in networking adapters."""
    dummy_repo = tmp_path / "mock_src"
    dummy_repo.mkdir()
    (dummy_repo / "data").mkdir()

    bad_code = """
from urllib.request import Request

def submit():
    req = Request("https://api.example.com/order", method="POST")
    return req
"""
    (dummy_repo / "data" / "order_sender.py").write_text(bad_code, encoding="utf-8")

    verifier = SafetyVerifier(root_path=dummy_repo)
    result = verifier.verify_all()

    assert not result.passed
    assert any("POST" in v for v in result.violations)


def test_safety_verifier_detects_prohibited_cli_flags(tmp_path):
    """Verify safety verifier detects --live or --real flags in cli.py."""
    dummy_repo = tmp_path / "mock_src"
    dummy_repo.mkdir()

    cli_code = """
import argparse
p = argparse.ArgumentParser()
p.add_argument("--live", action="store_true")
"""
    (dummy_repo / "cli.py").write_text(cli_code, encoding="utf-8")

    verifier = SafetyVerifier(root_path=dummy_repo)
    result = verifier.verify_all()

    assert not result.passed
    assert any("--live" in v for v in result.violations)
