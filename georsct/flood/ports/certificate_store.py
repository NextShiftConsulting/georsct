"""Port: certificate persistence.

Contract for storing/retrieving readiness certificates.
Adapter impls: S3 JSON, PostGIS table, local file.
"""

from abc import ABC, abstractmethod
from typing import Optional

from georsct.domain.certificate import ReadinessCertificate


class CertificateStore(ABC):
    """Abstract port for certificate storage."""

    @abstractmethod
    def store(
        self,
        scenario_id: str,
        unit_id: str,
        certificate: ReadinessCertificate,
    ) -> None:
        """Persist a certificate for a (scenario, unit) pair."""

    @abstractmethod
    def load(
        self,
        scenario_id: str,
        unit_id: str,
    ) -> Optional[ReadinessCertificate]:
        """Retrieve a certificate. Returns None if not found."""

    @abstractmethod
    def load_scenario(
        self,
        scenario_id: str,
    ) -> dict[str, ReadinessCertificate]:
        """Retrieve all certificates for a scenario. Keyed by unit_id."""
