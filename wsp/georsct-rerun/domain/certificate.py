"""Readiness certificate data structures.

Pure domain objects -- no I/O, no S3, no SQL.
Decision vocabulary per ADR-029/034.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PublicDecision(str, Enum):
    """Public (certificate layer) decision vocabulary."""

    EXECUTE = "EXECUTE"
    CAUTION = "CAUTION"
    REFUSE = "REFUSE"


class InternalDecision(str, Enum):
    """Internal (gatekeeper layer) decision vocabulary."""

    EXECUTE = "EXECUTE"
    REJECT = "REJECT"
    BLOCK = "BLOCK"
    RE_ENCODE = "RE_ENCODE"
    REPAIR = "REPAIR"
    WARN = "WARN"
    FALLBACK = "FALLBACK"


@dataclass(frozen=True)
class ReadinessCertificate:
    """RSCT readiness certificate for a spatial unit."""

    R: float
    S: float
    N: float
    kappa_compat: float

    kappa_H: Optional[float] = None
    kappa_L: Optional[float] = None
    kappa_interface: Optional[float] = None

    decision: PublicDecision = PublicDecision.REFUSE
    evidence_warnings: list[str] = field(default_factory=list)

    @property
    def simplex_valid(self) -> bool:
        return abs(self.R + self.S + self.N - 1.0) < 1e-6

    def to_yrsn_dict(self) -> dict:
        """Serialize for yrsn interoperability."""
        return {
            "R": self.R,
            "S": self.S,
            "N": self.N,
            "kappa_compat": self.kappa_compat,
            "kappa_H": self.kappa_H,
            "kappa_L": self.kappa_L,
            "kappa_interface": self.kappa_interface,
            "decision": self.decision.value,
        }

    @classmethod
    def from_yrsn_dict(cls, d: dict) -> "ReadinessCertificate":
        """Deserialize from yrsn format."""
        return cls(
            R=d["R"],
            S=d["S"],
            N=d["N"],
            kappa_compat=d["kappa_compat"],
            kappa_H=d.get("kappa_H"),
            kappa_L=d.get("kappa_L"),
            kappa_interface=d.get("kappa_interface"),
            decision=PublicDecision(d.get("decision", "REFUSE")),
        )
