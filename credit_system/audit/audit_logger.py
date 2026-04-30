"""
Tamper-Evident Audit Logger
==============================
Implements SHA-256 hash chaining — each audit entry includes the hash
of the previous entry. Any post-hoc modification breaks the chain
and is detected by verify_chain().
"""
import hashlib, json, time, uuid, os
from pathlib import Path
from typing import Optional

AUDIT_LOG_PATH = Path(os.path.join(os.path.dirname(__file__), "audit_chain.jsonl"))


def _get_last_hash() -> str:
    """Return hash of the last block, or 'GENESIS' if log is empty."""
    if not AUDIT_LOG_PATH.exists() or AUDIT_LOG_PATH.stat().st_size == 0:
        return "GENESIS"
    with open(AUDIT_LOG_PATH, "rb") as f:
        # Read last non-empty line
        lines = f.read().decode("utf-8").strip().split("\n")
    last = json.loads(lines[-1])
    return last["block_hash"]


def log_decision(
    application_id: str,
    inputs: dict,
    decision: dict,
    shap_output: dict,
    officer_id: Optional[str] = None,
) -> dict:
    """
    Append a tamper-evident audit entry to the chain.

    Parameters
    ----------
    application_id : Unique ID for this loan application
    inputs         : Raw feature dict sent to the model
    decision       : {"approved": bool, "probability": float, "threshold": float}
    shap_output    : Full SHAP explanation dict from SHAPExplainer.explain()
    officer_id     : Optional officer/agent ID who initiated the request

    Returns the complete audit block including its hash.
    """
    prev_hash = _get_last_hash()

    entry = {
        "block_id":        str(uuid.uuid4()),
        "timestamp_utc":   time.time(),
        "timestamp_iso":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "application_id":  application_id,
        "officer_id":      officer_id or "system",
        "inputs":          inputs,
        "decision":        decision,
        "shap_output":     shap_output,
        "prev_hash":       prev_hash,
    }

    # Compute hash BEFORE adding block_hash key
    entry_str  = json.dumps(entry, sort_keys=True, ensure_ascii=True)
    block_hash = hashlib.sha256(entry_str.encode("utf-8")).hexdigest()
    entry["block_hash"] = block_hash

    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")

    return entry


def verify_chain() -> dict:
    """
    Walk every block in the audit log and verify the SHA-256 chain.
    Returns {"valid": bool, "total_blocks": int, "broken_at_block": str | None}
    """
    if not AUDIT_LOG_PATH.exists() or AUDIT_LOG_PATH.stat().st_size == 0:
        return {"valid": True, "total_blocks": 0, "broken_at_block": None}

    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    prev_hash = "GENESIS"
    for idx, line in enumerate(lines):
        entry = json.loads(line)
        stored_hash = entry.pop("block_hash")

        # Verify prev_hash linkage
        if entry["prev_hash"] != prev_hash:
            return {
                "valid":           False,
                "total_blocks":    len(lines),
                "broken_at_block": entry["block_id"],
                "reason":          f"prev_hash mismatch at block {idx + 1}",
            }

        # Recompute hash
        recomputed = hashlib.sha256(
            json.dumps(entry, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()

        if recomputed != stored_hash:
            return {
                "valid":           False,
                "total_blocks":    len(lines),
                "broken_at_block": entry["block_id"],
                "reason":          f"Hash mismatch at block {idx + 1} — content was modified",
            }

        prev_hash = stored_hash

    return {"valid": True, "total_blocks": len(lines), "broken_at_block": None}


def get_decision(application_id: str) -> Optional[dict]:
    """Retrieve a single audit entry by application_id."""
    if not AUDIT_LOG_PATH.exists():
        return None
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("application_id") == application_id:
                return entry
    return None


def get_recent_decisions(limit: int = 50) -> list:
    """Return the most recent N audit entries."""
    if not AUDIT_LOG_PATH.exists():
        return []
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    recent = lines[-limit:]
    return [json.loads(l) for l in reversed(recent)]
