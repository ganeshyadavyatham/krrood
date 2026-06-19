"""
Step 6 — The warning object.

A small, printable record of an out-of-distribution detection: which rule
node flagged it, the log-likelihood that triggered it, and a human-readable
reason. This is what gives your system "traceability" (the assignment's word).
"""

from dataclasses import dataclass
from typing_extensions import Optional


@dataclass
class UnfamiliarSampleWarning:
    node_name: str
    log_p: Optional[float]
    reason: str

    def __str__(self) -> str:
        return f"UnfamiliarSampleWarning at '{self.node_name}': {self.reason}"
