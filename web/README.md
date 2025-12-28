# Gomoku Zero Web

This is a Next.js + Tailwind demo UI for playing the 9x9 AI over WebSocket.

## Local development

```bash
npm install
npm run dev
```

Then open `http://localhost:3000` and point the UI at your backend WebSocket
(`wss://smokingmouse-gomokuzero.hf.space/ws/play` by default).

## Static export (GitHub Pages)

```bash
npm run build
```

If the site is hosted at a subpath (for example `/GomokuZero` on GitHub
Pages), set:

```bash
NEXT_PUBLIC_BASE_PATH=/GomokuZero npm run build
```

## GitHub Pages deployment

A workflow is included at `.github/workflows/gh-pages.yml`. It publishes the
static export to GitHub Pages on every push to `main`.

Steps:

1. In your GitHub repo, go to Settings → Pages.
2. Under Build and deployment, select GitHub Actions.
3. Push to `main` and wait for the workflow to finish.

The workflow sets `NEXT_PUBLIC_BASE_PATH=/<repo-name>` automatically for the
GitHub Pages subpath.
