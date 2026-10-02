"""Autoridad de aprobación de órdenes."""
from __future__ import annotations

import dataclasses
import hashlib
import hmac
import secrets
import time

from .exceptions import UnapprovedOrderError
from .types import ApprovedOrder


class ApprovalAuthority:
    """Firma y verifica ``ApprovedOrder``."""

    def __init__(self, ttl_seconds: float = 90.0, clock=time.time) -> None:
        self._key = secrets.token_bytes(32)
        self.ttl_seconds = ttl_seconds
        self._clock = clock

    @staticmethod
    def _payload(o: ApprovedOrder) -> bytes:
        return (
            f"{o.proposal_id}|{o.symbol}|{int(o.direction)}|{o.volume:.8f}|{o.price:.8f}|"
            f"{o.sl:.8f}|{o.tp:.8f}|{o.issued_at:.3f}"
        ).encode()

    def sign(self, order: ApprovedOrder) -> ApprovedOrder:
        token = hmac.new(self._key, self._payload(order), hashlib.sha256).hexdigest()
        return dataclasses.replace(order, token=token)

    def verify(self, order: ApprovedOrder) -> None:
        if not isinstance(order, ApprovedOrder) or not order.token:
            raise UnapprovedOrderError("Orden sin firma del agente de riesgo")
        expected = hmac.new(self._key, self._payload(order), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, order.token):
            raise UnapprovedOrderError("Firma de aprobación inválida (orden alterada o no emitida por RiskAgent)")
        age = self._clock() - order.issued_at
        if age > self.ttl_seconds:
            raise UnapprovedOrderError(f"Aprobación caducada ({age:.0f}s > {self.ttl_seconds:.0f}s)")


# Autoridad compartida por defecto (una por proceso).
DEFAULT_AUTHORITY = ApprovalAuthority()
