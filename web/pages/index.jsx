import { useEffect, useRef, useState } from "react";

const DEFAULT_WS =
  process.env.NEXT_PUBLIC_WS_URL ||
  "wss://smokingmouse-gomokuzero.hf.space/ws/play";

const makeEmptyBoard = (size) =>
  Array.from({ length: size }, () => Array.from({ length: size }, () => 0));

export default function Home() {
  const [wsUrl, setWsUrl] = useState(DEFAULT_WS);
  const [connected, setConnected] = useState(false);
  const [humanPlayer, setHumanPlayer] = useState(2);
  const [board, setBoard] = useState(() => makeEmptyBoard(9));
  const [currentPlayer, setCurrentPlayer] = useState(1);
  const [winner, setWinner] = useState(0);
  const [done, setDone] = useState(false);
  const [status, setStatus] = useState("disconnected");
  const [lastMove, setLastMove] = useState(null);
  const [error, setError] = useState("");
  const socketRef = useRef(null);

  useEffect(() => {
    const stored = window.localStorage.getItem("gomoku_ws_url");
    if (stored) {
      setWsUrl(stored);
    }
  }, []);

  const boardSize = board.length;
  const canPlay = connected && !done && currentPlayer === humanPlayer;
  const statusLabel = done
    ? winner === 0
      ? "Draw"
      : winner === humanPlayer
      ? "You win"
      : "AI wins"
    : status === "ai_thinking"
    ? "AI thinking..."
    : canPlay
    ? "Your move"
    : "Waiting for AI";

  const connect = () => {
    setError("");
    if (!wsUrl.startsWith("ws")) {
      setError("WebSocket URL must start with ws:// or wss://");
      return;
    }
    window.localStorage.setItem("gomoku_ws_url", wsUrl);
    const socket = new WebSocket(wsUrl);
    socketRef.current = socket;
    socket.onopen = () => {
      setConnected(true);
      setStatus("connected");
      socket.send(
        JSON.stringify({
          action: "new_game",
          human_player: humanPlayer
        })
      );
    };
    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === "game_state") {
          setBoard(message.board);
          setCurrentPlayer(message.current_player);
          setWinner(message.winner);
          setDone(message.done);
          setLastMove(message.last_move || null);
          setStatus("connected");
          return;
        }
        if (message.type === "status") {
          setStatus(message.message);
          return;
        }
        if (message.type === "error") {
          setError(message.message);
        }
      } catch (err) {
        setError("Failed to parse server message.");
      }
    };
    socket.onclose = () => {
      setConnected(false);
      setStatus("disconnected");
      socketRef.current = null;
    };
  };

  const disconnect = () => {
    if (socketRef.current) {
      socketRef.current.close();
    }
  };

  const requestNewGame = () => {
    setError("");
    setDone(false);
    setWinner(0);
    setStatus("connected");
    setLastMove(null);
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(
        JSON.stringify({
          action: "new_game",
          human_player: humanPlayer
        })
      );
    }
  };

  const handleMove = (row, col) => {
    if (!canPlay || !socketRef.current) {
      return;
    }
    socketRef.current.send(
      JSON.stringify({
        action: "move",
        x: row,
        y: col
      })
    );
  };

  return (
    <div className="min-h-screen px-4 py-8 sm:px-6 sm:py-10">
      <div className="mx-auto grid w-full max-w-6xl gap-8 lg:grid-cols-[320px,1fr]">
        <section className="flex flex-col gap-6 animate-float-in">
          <div className="rounded-2xl bg-white/70 p-6 shadow-lg shadow-black/5">
            <p className="font-mono text-xs uppercase tracking-[0.35em] text-emerald-700">
              Gomoku Zero Arena
            </p>
            <h1 className="mt-3 text-3xl font-semibold text-ink">
              Challenge the 9x9 AI
            </h1>
            <p className="mt-3 text-sm text-slate-700">
              Connect to your FastAPI server and play against the latest ZeroMCTS
              model. The board updates in real time via WebSocket.
            </p>
          </div>

          <div className="rounded-2xl border border-black/10 bg-white/80 p-5 shadow-lg shadow-black/5">
            <div className="flex items-center justify-between">
              <p className="text-sm font-semibold text-slate-900">Connection</p>
              <span
                className={`rounded-full px-2 py-1 text-xs font-semibold ${
                  connected
                    ? "bg-emerald-100 text-emerald-700"
                    : "bg-rose-100 text-rose-700"
                }`}
              >
                {connected ? "Online" : "Offline"}
              </span>
            </div>

            <label className="mt-4 block text-xs font-semibold uppercase tracking-[0.25em] text-slate-500">
              WebSocket URL
            </label>
            <input
              className="mt-2 w-full rounded-xl border border-black/10 bg-white px-3 py-2 text-sm shadow-inner"
              value={wsUrl}
              onChange={(event) => setWsUrl(event.target.value)}
              placeholder={DEFAULT_WS}
            />

            <div className="mt-4 flex gap-2">
              <button
                className="flex-1 rounded-xl bg-ink px-3 py-2 text-sm font-semibold text-white transition hover:bg-black"
                onClick={connect}
                disabled={connected}
              >
                Connect
              </button>
              <button
                className="flex-1 rounded-xl border border-black/10 bg-white px-3 py-2 text-sm font-semibold text-ink transition hover:border-black/30"
                onClick={disconnect}
                disabled={!connected}
              >
                Disconnect
              </button>
            </div>
          </div>

          <div className="rounded-2xl border border-black/10 bg-white/80 p-5 shadow-lg shadow-black/5">
            <p className="text-sm font-semibold text-slate-900">Match Settings</p>
            <div className="mt-4 flex items-center justify-between text-sm text-slate-700">
              <span>Play as</span>
              <div className="flex gap-2">
                <button
                  className={`rounded-full px-3 py-1 text-xs font-semibold transition ${
                    humanPlayer === 1
                      ? "bg-ink text-white"
                      : "bg-white text-slate-700"
                  }`}
                  onClick={() => setHumanPlayer(1)}
                >
                  Black
                </button>
                <button
                  className={`rounded-full px-3 py-1 text-xs font-semibold transition ${
                    humanPlayer === 2
                      ? "bg-ink text-white"
                      : "bg-white text-slate-700"
                  }`}
                  onClick={() => setHumanPlayer(2)}
                >
                  White
                </button>
              </div>
            </div>
            <button
              className="mt-4 w-full rounded-xl bg-ember px-3 py-2 text-sm font-semibold text-white transition hover:bg-orange-600"
              onClick={requestNewGame}
              disabled={!connected}
            >
              Start New Game
            </button>
            {error && (
              <p className="mt-3 text-xs font-semibold text-rose-600">
                {error}
              </p>
            )}
          </div>
        </section>

        <section className="flex flex-col gap-6 animate-float-in">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.3em] text-slate-500">
                Status
              </p>
              <p className="text-2xl font-semibold text-ink">{statusLabel}</p>
            </div>
            <div className="rounded-2xl border border-black/10 bg-white/80 px-4 py-2 text-sm text-slate-700">
              Current player: {currentPlayer === 1 ? "Black" : "White"}
            </div>
          </div>

          <div
            className="rounded-[24px] bg-white/70 p-4 sm:p-5 md:p-6 shadow-lg shadow-black/5"
            style={{
              width: "clamp(250px, 76vw, 320px)",
              margin: "0 auto"
            }}
          >
            <div
              className="board-surface grid gap-0 rounded-[18px] select-none"
              style={{
                gridTemplateColumns: `repeat(${boardSize}, minmax(0, 1fr))`,
                touchAction: "manipulation"
              }}
            >
              {board.map((row, rowIndex) =>
                row.map((cell, colIndex) => {
                  const isLast =
                    lastMove &&
                    lastMove.x === rowIndex &&
                    lastMove.y === colIndex;
                  return (
                    <button
                      key={`${rowIndex}-${colIndex}`}
                      className={`flex aspect-square items-center justify-center bg-transparent transition ${
                        canPlay ? "hover:bg-amber-100/70" : "cursor-default"
                      } ${isLast ? "ring-2 ring-ember/70" : ""}`}
                      onClick={() => handleMove(rowIndex, colIndex)}
                      disabled={!canPlay || cell !== 0}
                      aria-label={`Place at ${rowIndex}, ${colIndex}`}
                    >
                      {cell !== 0 && (
                        <span
                          className={`stone h-5 w-5 sm:h-6 sm:w-6 ${
                            cell === 1 ? "stone-black" : "stone-white"
                          }`}
                        />
                      )}
                    </button>
                  );
                })
              )}
            </div>
          </div>

          <div className="rounded-2xl border border-black/10 bg-white/80 p-5 text-sm text-slate-700">
            <p className="font-semibold text-slate-900">How to play</p>
            <p className="mt-2">
              Connect to the server, pick your color, and click a square to place
              a stone. The AI responds automatically after each move.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
