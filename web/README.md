# Gomoku Zero Web

This is a Next.js + Tailwind demo UI for playing the 9x9 AI over WebSocket.

## Local development

```bash
npm install
npm run dev
```

Then open `http://localhost:3000` and point the UI at your backend WebSocket
(`ws://localhost:8000/ws/play` by default).

## Static export (GitHub Pages)

```bash
npm run build
npm run export
```

If the site is hosted at a subpath (for example `/GomokuZero` on GitHub
Pages), set:

```bash
NEXT_PUBLIC_BASE_PATH=/GomokuZero npm run build
NEXT_PUBLIC_BASE_PATH=/GomokuZero npm run export
```
