"""Excepciones del sistema."""


class MT5AGIError(Exception):
    """Base de todas las excepciones del sistema."""


class MT5ConnectionError(MT5AGIError):
    """No fue posible conectar / mantener la conexión con MetaTrader 5."""


class RealAccountBlockedError(AssertionError, MT5AGIError):
    """Se detectó una cuenta REAL. Hereda de ``AssertionError`` a propósito."""


class UnapprovedOrderError(MT5AGIError):
    """Se intentó enviar una orden que no fue aprobada por el agente de riesgo."""


class OrderRejectedError(MT5AGIError):
    """El broker rechazó la orden."""


class DataUnavailableError(MT5AGIError):
    """No hay datos de mercado disponibles para la consulta."""
