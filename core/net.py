"""HTTP común: confía en el almacén de certificados del sistema operativo y usa cabeceras de navegador."""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)
_LOCK = threading.Lock()
_INJECTED = False

BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "application/rss+xml, application/xml, application/json;q=0.9, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def ensure_os_trust() -> bool:
    """Inyecta ``truststore`` una sola vez."""
    global _INJECTED
    with _LOCK:
        if _INJECTED:
            return True
        try:
            import truststore

            truststore.inject_into_ssl()
            _INJECTED = True
        except Exception as exc:
            log.warning("truststore no disponible (%s): se usará el bundle de certifi", exc)
    return _INJECTED


def http_get(url: str, timeout: float = 10.0):
    """GET con verificación TLS contra el almacén del SO. Lanza ``requests.HTTPError`` si el estado no es 2xx."""
    import requests

    ensure_os_trust()
    r = requests.get(url, timeout=timeout, headers=BROWSER_HEADERS)
    r.raise_for_status()
    return r
