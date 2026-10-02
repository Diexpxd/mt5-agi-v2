"""Utilidades de tiempo: sesiones de mercado."""
from __future__ import annotations

from datetime import datetime


def session_name(dt: datetime) -> str:
    """Sesión de mercado por hora UTC: asia (00-07), london (07-13), overlap (13-16), newyork (16-21), offhours."""
    h = dt.hour
    if 0 <= h < 7:
        return "asia"
    if 7 <= h < 13:
        return "london"
    if 13 <= h < 16:
        return "overlap"
    if 16 <= h < 21:
        return "newyork"
    return "offhours"
