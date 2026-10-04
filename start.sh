#!/usr/bin/env bash
# Website lokal testen (macOS/Linux). Für die Online-Nutzung wird nichts davon gebraucht.
cd "$(dirname "$0")"
( sleep 1; (command -v open >/dev/null && open http://127.0.0.1:8000) || (command -v xdg-open >/dev/null && xdg-open http://127.0.0.1:8000) ) >/dev/null 2>&1 &
exec python3 tools/serve.py 8000
