# Web Demo Deployment Notes

This repository includes a FastAPI WebSocket backend and a static Next.js UI.

## Backend (FastAPI)

Run locally or on a server:

```bash
uvicorn gomoku.app:app --host 0.0.0.0 --port 8000
```

Configuration (environment variables):
- `GOMOKU_MODEL_PATH` (default: `gomoku/models/gomoku_zero_9_plus_pro_max/policy_step_199000.pth`)
- `GOMOKU_MCTS_ITERS` (default: `400`)
- `GOMOKU_MCTS_PUCT` (default: `2.0`)
- `GOMOKU_AI_WORKERS` (default: `2`)
- `GOMOKU_CORS_ORIGINS` (default: `*`, comma-separated list)

The WebSocket endpoint is `ws://<host>:8000/ws/play`. For production, terminate TLS
and use `wss://` from the browser.

## Frontend (Next.js)

The UI lives under `web/`. It is designed for static export and GitHub Pages.
Build instructions are in `web/README.md`.

After exporting, serve the `web/out/` directory from GitHub Pages or any static host.
Update the WebSocket URL in the UI to point at your backend.

## Render (recommended free backend host)

1. Create a new Web Service from your GitHub repo.
2. Environment: Python 3.12.
3. Build command:
   - `pip install -e .`
4. Start command:
   - `uvicorn gomoku.app:app --host 0.0.0.0 --port $PORT`
5. Add environment variables if needed:
   - `GOMOKU_MODEL_PATH=gomoku/models/gomoku_zero_9_plus_pro_max/policy_step_199000.pth`
   - `GOMOKU_MCTS_ITERS=400`
   - `GOMOKU_AI_WORKERS=2`

Render will give you a public URL like `https://your-app.onrender.com`.
Use `wss://your-app.onrender.com/ws/play` in the frontend.

## GitHub Pages (frontend)

Set the default backend URL at build time:

```bash
NEXT_PUBLIC_WS_URL=wss://your-app.onrender.com/ws/play \
NEXT_PUBLIC_BASE_PATH=/GomokuZero \
npm run build
NEXT_PUBLIC_WS_URL=wss://your-app.onrender.com/ws/play \
NEXT_PUBLIC_BASE_PATH=/GomokuZero \
npm run export
```
