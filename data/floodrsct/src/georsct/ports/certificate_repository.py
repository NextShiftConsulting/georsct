"""Port: certificate persistence for GeoRSCT.

Reusable across hazard types (flood, earthquake, wildfire).
"""

from abc import ABC, abstractmethod
from typing import Optional

from georsct.domain.certificate import ReadinessCertificate


class CertificateRepository(ABC):

    @abstractmethod
    def save(self, key: str, cert: ReadinessCertificate) -> None: ...

    @abstractmethod
    def load(self, key: str) -> Optional[ReadinessCertificate]: ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]: ...
