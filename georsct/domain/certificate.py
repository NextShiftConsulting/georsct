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
    """Internal (gatekeeper layer) decision vocabulary per ADR-029/034.

    This is the correct vocabulary for DGM routing decisions.  Earlier code
    (compute_dgm_routing.py) imported yrsn MorphType and injected RE_ENCODE /
    FALLBACK via hasattr guards -- those values don't exist in MorphType because
    MorphType describes graph morphisms, not gatekeeper outcomes.

    Three-layer decision architecture:
        MorphType (yrsn dgm_unified)  -- what graph transformation to apply
        InternalDecision (here)       -- what the gatekeeper decides to do
        PublicDecision (here)         -- what the end user sees

    DGM routing produces InternalDecision; the certificate projects it to
    PublicDecision at the API boundary.
    """

    EXECUTE = "EXECUTE"
    REJECT = "REJECT"
    BLOCK = "BLOCK"
    RE_ENCODE = "RE_ENCODE"
    REPAIR = "REPAIR"
    WARN = "WARN"
    FALLBACK = "FALLBACK"


@dataclass(frozen=True)
class ReadinessCertificate:
    """RSCT readiness certificate for a spatial unit.

    Field names per ADR-034:
      S_sup (not bare S), kappa_compat (not kappa_gate),
      public_decision (not bare decision).
    """

    R: float
    S_sup: float
    N: float
    kappa_compat: float

    kappa_H: Optional[float] = None
    kappa_L: Optional[float] = None
    kappa_interface: Optional[float] = None

    public_decision: PublicDecision = PublicDecision.REFUSE
    evidence_warnings: list[str] = field(default_factory=list)

    @property
    def simplex_valid(self) -> bool:
        return abs(self.R + self.S_sup + self.N - 1.0) < 1e-6

    def to_yrsn_dict(self) -> dict:
        """Serialize for yrsn interoperability."""
        return {
            "R": self.R,
            "S_sup": self.S_sup,
            "N": self.N,
            "kappa_compat": self.kappa_compat,
            "kappa_H": self.kappa_H,
            "kappa_L": self.kappa_L,
            "kappa_interface": self.kappa_interface,
            "public_decision": self.public_decision.value,
        }

    @classmethod
    def from_yrsn_dict(cls, d: dict) -> "ReadinessCertificate":
        """Deserialize from yrsn format."""
        return cls(
            R=d["R"],
            S_sup=d.get("S_sup", d.get("S", 0.0)),
            N=d["N"],
            kappa_compat=d.get("kappa_compat", d.get("kappa_gate", 0.0)),
            kappa_H=d.get("kappa_H"),
            kappa_L=d.get("kappa_L"),
            kappa_interface=d.get("kappa_interface"),
            public_decision=PublicDecision(
                d.get("public_decision", d.get("decision", "REFUSE"))
            ),
        )
