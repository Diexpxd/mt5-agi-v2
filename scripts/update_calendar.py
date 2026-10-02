"""Actualiza data/economic_calendar.json con el calendario real de la semana (Forex Factory / faireconomy)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from _common import ROOT

from agents.economic_calendar import EconomicCalendar, generate_sample_calendar, refresh_from_web   # noqa: E402
from config.settings import load_settings                                                            # noqa: E402

s = load_settings()
path = s.data_dir / "economic_calendar.json"
cal = EconomicCalendar.from_json(path)
try:
    events = refresh_from_web()
    cal.events = [e for e in cal.events if e.time > datetime.now(timezone.utc) - timedelta(days=30)]   # poda lo antiguo
    cal.merge(events)
    cal.to_json(path)
    hi = [e for e in events if e.impact == "high"]
    print(f"✅ {len(events)} eventos descargados ({len(hi)} de alto impacto) → {path}")
    for e in sorted(hi, key=lambda e: e.time)[:8]:
        print(f"   {e.time:%a %d %H:%M} UTC  {e.currency}  {e.title}")
except Exception as exc:
    print(f"⚠️ No se pudo descargar el calendario: {exc}")
    if not cal.events:
        now = datetime.now(timezone.utc)
        cal = EconomicCalendar(generate_sample_calendar(now - timedelta(days=7), now + timedelta(days=60)))
        cal.to_json(path, sample=True)
        print(f"   Generado calendario de EJEMPLO ({len(cal.events)} eventos, fechas aproximadas) → {path}")
