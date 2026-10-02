# MT5 AGI V2

Sistema multiagente de trading automático sobre MetaTrader 5. Un orquestador coordina cuatro agentes (técnico, fundamental, riesgo e interfaz), con una memoria vectorial que aprende de las operaciones cerradas y un backtester propio. Solo funciona con cuentas demo.

Es un proyecto de investigación y aprendizaje. No es asesoramiento financiero.

## Cómo funciona

Por cada vela cerrada el orquestador hace este recorrido:

1. **Agente técnico**: estructura de mercado, order blocks, fair value gaps y barridos de liquidez, más un modelo LSTM/Transformer (PyTorch) que solo pesa si supera su validación.
2. **Agente fundamental**: titulares RSS y calendario económico; el sentimiento sale de un LLM (Gemini) o, sin clave, de un léxico financiero local.
3. **Memoria**: ChromaDB y SQLite guardan las operaciones cerradas y ajustan el peso de riesgo según el contexto (por ejemplo, un par que pierde durante NFP).
4. **Agente de riesgo**: Kelly fraccional dinámico, VaR, exposición por divisa y cortes por pérdida diaria o drawdown. Ninguna orden llega al broker sin su firma HMAC, que `send_order` verifica.
5. **Ejecución**: modo `paper` (fills simulados con precios reales) o `demo` (órdenes a una cuenta demo de MT5).

El agente de interfaz atiende un bot de Telegram en lenguaje natural (español e inglés): PnL, resumen de mercado, gráficos y comandos de administración.

```
agents/        técnico, fundamental, riesgo, interfaz y orquestador
core/          conexión a MT5, ejecución, firma de órdenes, simulador de MT5
memory/        diario de operaciones y almacén vectorial
backtesting/   motor, métricas y carga de datos
bot/           bot de Telegram y gráficos
scripts/       arranque, backtest, entrenamiento, descarga de histórico
tests/         185 tests con pytest
```

## Instalación

Requiere Python 3.10 o superior (probado en 3.12).

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Todas las variables de `.env` son opcionales. Sin claves el sistema usa un MT5 simulado y el léxico local.

## Uso

Prueba rápida sin MT5 ni claves:

```
python scripts/run_system.py --mock --console --fast
```

Con terminal MT5 y cuenta demo abierta:

```
python scripts/check_setup.py
python scripts/run_system.py                 # paper
python scripts/run_system.py --mode demo     # cuenta demo de MT5
```

Backtest e histórico:

```
python scripts/download_history.py --bars 60000
python scripts/run_backtest.py --source csv --train-frac 0.5
python scripts/train_models.py --bars 30000 --arch transformer
```

Tests:

```
python -m pytest
```

## Límites de riesgo por defecto

Se editan en `config/settings.py` (`RiskLimits`): 1 % de riesgo máximo por operación, Kelly x0.25, riesgo abierto total 4 %, apalancamiento 10x, 6x por divisa, VaR 95 % de 1 día hasta 3 % del equity, corte diario de -3 % y corte por drawdown de 10 %. Hay silencio de 30 minutos alrededor de noticias de alto impacto.

La conexión se bloquea si detecta una cuenta real, de concurso o un servidor con "live" o "real" en el nombre. La comprobación se repite al conectar, en cada reconexión y antes de cada orden.

## Estado y limitaciones

- No he podido validarlo contra un terminal MT5 real: la conexión, el envío de órdenes y la calibración horaria solo se probaron contra el simulador incluido.
- No hay ventaja demostrada. Los backtests se hicieron solo con datos sintéticos y el resultado es indistinguible del azar tras costes. Sirve para investigar, no como sistema rentable.
- El LLM (Gemini) y el bot de Telegram se probaron con clientes simulados, no con claves reales.
- El sentimiento léxico es simple y solo en inglés.
