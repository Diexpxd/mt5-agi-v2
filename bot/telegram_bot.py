"""Chatbot de Telegram (telebot / AsyncTeleBot) sobre el Agente de Interfaz."""
from __future__ import annotations

import asyncio
import io
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

from agents.interface_agent import InterfaceAgent, Reply
from config.settings import Settings

log = logging.getLogger(__name__)
MAX_MSG = 4000                                              # Telegram admite 4096 caracteres por mensaje


class Sender(Protocol):
    async def send_text(self, chat_id: int, text: str, buttons: Optional[List[List[Tuple[str, str]]]] = None) -> None: ...
    async def send_photo(self, chat_id: int, png: bytes, caption: str = "") -> None: ...


def split_text(text: str, limit: int = MAX_MSG) -> List[str]:
    """Parte un texto largo en mensajes ≤ ``limit`` respetando saltos de línea."""
    if len(text) <= limit:
        return [text]
    out, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > limit and cur:
            out.append(cur)
            cur = ""
        while len(line) > limit:                             # línea gigante sin saltos
            out.append(line[:limit])
            line = line[limit:]
        cur = f"{cur}\n{line}" if cur else line
    if cur:
        out.append(cur)
    return out


async def deliver(sender: Sender, chat_id: int, reply: Reply) -> None:
    """Envía una ``Reply``: primero las imágenes (la primera con el texto como pie si cabe), luego texto y botones."""
    text = reply.text or ""
    photos = list(reply.images)
    text_sent = False
    if photos:
        caption = text if len(text) <= 1000 and len(photos) == 1 and not reply.buttons else ""
        for i, png in enumerate(photos):
            await sender.send_photo(chat_id, png, caption if i == 0 else "")
        text_sent = bool(caption)
    if not text_sent and (text or reply.buttons):
        chunks = split_text(text or "Menú")
        for i, chunk in enumerate(chunks):
            await sender.send_text(chat_id, chunk, reply.buttons if i == len(chunks) - 1 else None)


class TradingBot:
    """Lógica del bot desacoplada del transporte (testeable sin Telegram)."""

    def __init__(self, settings: Settings, interface: InterfaceAgent, allowed_ids: Iterable[int] = ()) -> None:
        self.settings = settings
        self.interface = interface
        self.admins = set(settings.telegram_admin_ids)
        self.viewers = set(allowed_ids) | set(int(x) for x in os.getenv("TELEGRAM_ALLOWED_IDS", "").split(",") if x.strip().lstrip("-").isdigit())

    def role(self, user_id: int) -> str:
        return "admin" if user_id in self.admins else "viewer" if user_id in self.viewers else "none"

    def _denied(self, user_id: int) -> Reply:
        return Reply(f"🔒 Acceso no autorizado.\nTu id de Telegram es {user_id}. Añádelo a TELEGRAM_ADMIN_IDS (o TELEGRAM_ALLOWED_IDS "
                     "para solo lectura) en el archivo .env y reinicia el sistema.")

    def on_text(self, user_id: int, text: str) -> Reply:
        role = self.role(user_id)
        if role == "none":
            return self._denied(user_id)
        return self.interface.handle(text, is_admin=(role == "admin"))

    def on_callback(self, user_id: int, data: str) -> Reply:
        role = self.role(user_id)
        if role == "none":
            return self._denied(user_id)
        return self.interface.handle_callback(data, is_admin=(role == "admin"))

    def alert_text(self, kind: str, payload: Dict[str, Any]) -> Optional[str]:
        return self.interface.format_event(kind, payload)


class TelebotSender:
    """Transporte real: ``telebot.async_telebot.AsyncTeleBot``."""

    def __init__(self, bot) -> None:
        self.bot = bot

    @staticmethod
    def _markup(buttons):
        if not buttons:
            return None
        from telebot import types

        m = types.InlineKeyboardMarkup()
        for row in buttons:
            m.row(*[types.InlineKeyboardButton(text=t, callback_data=d) for t, d in row])
        return m

    async def send_text(self, chat_id, text, buttons=None):
        await self.bot.send_message(chat_id, text, reply_markup=self._markup(buttons))

    async def send_photo(self, chat_id, png, caption=""):
        await self.bot.send_photo(chat_id, io.BytesIO(png), caption=caption or None)


class ConsoleSender:
    """Transporte de consola para desarrollo sin token: imprime y guarda los PNG en ``out_dir``."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.n = 0
        out_dir.mkdir(parents=True, exist_ok=True)

    async def send_text(self, chat_id, text, buttons=None):
        print(f"\n[bot] {text}")
        if buttons:
            print("      botones: " + " | ".join(f"{t} ({d})" for row in buttons for t, d in row))

    async def send_photo(self, chat_id, png, caption=""):
        self.n += 1
        path = self.out_dir / f"chart_{self.n:03d}.png"
        path.write_bytes(png)
        print(f"\n[bot] 🖼  gráfico guardado en {path}" + (f"\n{caption}" if caption else ""))


def build_async_bot(settings: Settings, trading_bot: TradingBot):
    """Crea el ``AsyncTeleBot`` con todos los handlers registrados. Devuelve ``(bot, sender)``."""
    from telebot.async_telebot import AsyncTeleBot

    bot = AsyncTeleBot(settings.telegram_token, parse_mode=None)
    sender = TelebotSender(bot)
    loop_run = lambda fn, *a: asyncio.get_running_loop().run_in_executor(None, fn, *a)      # noqa: E731 (I/O y matplotlib fuera del loop)
    commands = ["start", "help", "menu", "pnl", "positions", "posiciones", "status", "estado", "market", "mercado", "chart",
                "grafico", "sentiment", "sentimiento", "signal", "senal", "risk", "riesgo", "lessons", "lecciones", "calendar",
                "calendario", "trades", "why", "porque", "equity", "pause", "pausa", "resume", "reanudar", "closeall", "cerrartodo"]

    @bot.message_handler(commands=commands)
    async def on_command(message):
        text = message.text or ""
        await bot.send_chat_action(message.chat.id, "typing")
        reply = await loop_run(trading_bot.on_text, message.from_user.id, text)
        await deliver(sender, message.chat.id, reply)

    @bot.message_handler(func=lambda m: True, content_types=["text"])
    async def on_text(message):
        await bot.send_chat_action(message.chat.id, "typing")
        reply = await loop_run(trading_bot.on_text, message.from_user.id, message.text or "")
        await deliver(sender, message.chat.id, reply)

    @bot.callback_query_handler(func=lambda c: True)
    async def on_callback(call):
        await bot.answer_callback_query(call.id)
        reply = await loop_run(trading_bot.on_callback, call.from_user.id, call.data or "")
        await deliver(sender, call.message.chat.id, reply)

    return bot, sender


class BotRunner:
    """Ejecuta el bot (Telegram real o consola) y reenvía las alertas del orquestador a los admins."""

    def __init__(self, settings: Settings, trading_bot: TradingBot, orchestrator) -> None:
        self.settings = settings
        self.tb = trading_bot
        self.orch = orchestrator
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._sender: Optional[Sender] = None
        orchestrator.subscribe(self._on_event)

    def _on_event(self, kind: str, payload: Dict[str, Any]) -> None:
        text = self.tb.alert_text(kind, payload)
        if not text or self._loop is None or self._sender is None:
            return
        for admin in self.tb.admins:
            asyncio.run_coroutine_threadsafe(self._sender.send_text(admin, text), self._loop)

    async def run_telegram(self) -> None:
        bot, sender = build_async_bot(self.settings, self.tb)
        self._loop, self._sender = asyncio.get_running_loop(), sender
        if not self.tb.admins:
            log.warning("TELEGRAM_ADMIN_IDS vacío: el bot solo mostrará a cada usuario su id. Configúralo en .env")
        log.info("Bot de Telegram en marcha (polling)")
        await bot.infinity_polling(skip_pending=True)

    async def run_console(self, user_id: int = 0) -> None:
        sender = ConsoleSender(self.settings.state_dir / "charts")
        self.tb.admins.add(user_id)
        loop = asyncio.get_running_loop()
        print("Modo consola del bot (sin token). Escribe una consulta, /help, o 'salir'.")
        while True:
            try:
                line = await loop.run_in_executor(None, input, "\ntú> ")
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip().lower() in ("salir", "exit", "quit"):
                break
            if line.strip():
                if line.strip().startswith("cb:"):
                    reply = await loop.run_in_executor(None, self.tb.on_callback, user_id, line.strip()[3:])
                else:
                    reply = await loop.run_in_executor(None, self.tb.on_text, user_id, line)
                await deliver(sender, 0, reply)

    def start_in_thread(self) -> threading.Thread:
        t = threading.Thread(target=lambda: asyncio.run(self.run_telegram()), name="telegram-bot", daemon=True)
        t.start()
        return t
