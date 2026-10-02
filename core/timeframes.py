"""Utilidades de marcos temporales (timeframes)."""
from __future__ import annotations

# minutos por marco temporal
TF_MINUTES = {
    "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M6": 6, "M10": 10, "M12": 12, "M15": 15, "M20": 20, "M30": 30,
    "H1": 60, "H2": 120, "H3": 180, "H4": 240, "H6": 360, "H8": 480, "H12": 720,
    "D1": 1440, "W1": 10080, "MN1": 43200,
}

TF_MT5_CONST = {
    "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M6": 6, "M10": 10, "M12": 12, "M15": 15, "M20": 20, "M30": 30,
    "H1": 16385, "H2": 16386, "H3": 16387, "H4": 16388, "H6": 16390, "H8": 16392, "H12": 16396,
    "D1": 16408, "W1": 32769, "MN1": 49153,
}

BARS_PER_YEAR_FX = 252 * 24 * 60  # minutos operativos por año (aprox., forex 24x5)


def tf_minutes(name: str) -> int:
    """Devuelve los minutos de un timeframe (p. ej. ``'H1' -> 60``)."""
    try:
        return TF_MINUTES[name.upper()]
    except KeyError as exc:
        raise ValueError(f"Timeframe desconocido: {name!r}") from exc


def bars_per_year(name: str) -> float:
    """Barras por año aproximadas (mercado 24x5) para anualizar métricas."""
    return BARS_PER_YEAR_FX / tf_minutes(name)


def bars_per_day(name: str) -> float:
    """Barras por día de trading (24h)."""
    return 1440.0 / tf_minutes(name)
