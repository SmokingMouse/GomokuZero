import { useEffect, useRef, useState } from "react";

const DEFAULT_WS =
  process.env.NEXT_PUBLIC_WS_URL ||
  "wss://smokingmouse-gomokuzero.hf.space/ws/play";

const deriveApiUrl = (wsUrl) => {
  if (!wsUrl) {
    return "";
  }
  try {
    const url = new URL(wsUrl);
    url.protocol = url.protocol === "wss:" ? "https:" : "http:";
    if (url.pathname.endsWith("/ws/play")) {
      url.pathname = url.pathname.slice(0, -"/ws/play".length) || "/";
    }
    url.hash = "";
    url.search = "";
    return url.toString().replace(/\/$/, "");
  } catch (err) {
    return "";
  }
};

const DEFAULT_API =
  process.env.NEXT_PUBLIC_API_URL ||
  deriveApiUrl(DEFAULT_WS) ||
  "http://localhost:8000";

const DEFAULT_BOARD_SIZE = Number(process.env.NEXT_PUBLIC_BOARD_SIZE || 15);
const makeEmptyBoard = (size) =>
  Array.from({ length: size }, () => Array.from({ length: size }, () => 0));

const cloneBoard = (board) => board.map((row) => row.slice());

const inferCurrentPlayer = (board) => {
  let black = 0;
  let white = 0;
  board.forEach((row) => {
    row.forEach((cell) => {
      if (cell === 1) {
        black += 1;
      } else if (cell === 2) {
        white += 1;
      }
    });
  });
  if (black === white) {
    return 1;
  }
  if (black === white + 1) {
    return 2;
  }
  return black > white ? 2 : 1;
};

export default function Home() {
  const [wsUrl, setWsUrl] = useState(DEFAULT_WS);
  const [apiUrl, setApiUrl] = useState(DEFAULT_API);
  const [connected, setConnected] = useState(false);
  const [matchMode, setMatchMode] = useState("human");
  const [humanPlayer, setHumanPlayer] = useState(2);
  const [board, setBoard] = useState(() => makeEmptyBoard(DEFAULT_BOARD_SIZE));
  const [currentPlayer, setCurrentPlayer] = useState(1);
  const [winner, setWinner] = useState(0);
  const [done, setDone] = useState(false);
  const [status, setStatus] = useState("disconnected");
  const [lastMove, setLastMove] = useState(null);
  const [error, setError] = useState("");
  const [aiDelayMs, setAiDelayMs] = useState(200);
  const [aiPaused, setAiPaused] = useState(false);
  const [usePreset, setUsePreset] = useState(false);
  const [editorBoard, setEditorBoard] = useState(() =>
    makeEmptyBoard(DEFAULT_BOARD_SIZE)
  );
  const [editorMode, setEditorMode] = useState("auto");
  const [editorBestActions, setEditorBestActions] = useState(() => new Set());
  const [editorHistory, setEditorHistory] = useState([]);
  const [editorCurrentPlayer, setEditorCurrentPlayer] = useState(1);
  const [editorLastAction, setEditorLastAction] = useState(-1);
  const [editorCursor, setEditorCursor] = useState({ row: 0, col: 0 });
  const [metaNote, setMetaNote] = useState("");
  const [metaTag, setMetaTag] = useState("");
  const [metaDifficulty, setMetaDifficulty] = useState(3);
  const [saveStatus, setSaveStatus] = useState("");
  const [saveError, setSaveError] = useState("");
  const [saving, setSaving] = useState(false);
  const [validationItems, setValidationItems] = useState([]);
  const [validationLoading, setValidationLoading] = useState(false);
  const [validationError, setValidationError] = useState("");
  const [selectedSampleId, setSelectedSampleId] = useState("");
  const [validationDifficultyFilter, setValidationDifficultyFilter] =
    useState("all");
  const socketRef = useRef(null);

  useEffect(() => {
    const stored = window.localStorage.getItem("gomoku_ws_url");
    if (stored) {
      setWsUrl(stored);
    }
    const storedApi = window.localStorage.getItem("gomoku_api_url");
    if (storedApi) {
      setApiUrl(storedApi);
    }
  }, []);

  const boardSize = board.length;
  const canPlay =
    connected &&
    !done &&
    matchMode === "human" &&
    currentPlayer === humanPlayer;
  const statusLabel = done
    ? winner === 0
      ? "Draw"
      : matchMode === "ai"
      ? `Winner: ${winner === 1 ? "Black" : "White"}`
      : winner === humanPlayer
      ? "You win"
      : "AI wins"
    : status === "ai_thinking"
    ? "AI thinking..."
    : status === "ai_paused"
    ? "AI paused"
    : status === "ai_stopped"
    ? "AI stopped"
    : matchMode === "ai"
    ? "AI vs AI running"
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
    window.localStorage.setItem("gomoku_api_url", apiUrl);
    const socket = new WebSocket(wsUrl);
    socketRef.current = socket;
    socket.onopen = () => {
      setConnected(true);
      setStatus("connected");
      startMatch(socket);
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
          if (message.message === "ai_paused") {
            setAiPaused(true);
          }
          if (message.message === "ai_resumed") {
            setAiPaused(false);
          }
          if (message.message === "ai_stopped") {
            setAiPaused(false);
          }
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
    setAiPaused(false);
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      startMatch(socketRef.current);
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

  const startMatch = (socket) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }
    let preset = undefined;
    if (usePreset) {
      if (editorBoard.length !== boardSize) {
        setError("Preset board size must match current match board size.");
        return;
      }
      preset = {
        board: editorBoard,
        current_player: editorCurrentPlayer,
        last_action: editorLastAction
      };
    }
    if (matchMode === "ai") {
      const pauseOnStart = aiPaused;
      socket.send(
        JSON.stringify({
          action: "ai_vs_ai",
          delay_ms: aiDelayMs,
          preset,
          pause_on_start: pauseOnStart
        })
      );
      if (!pauseOnStart) {
        setAiPaused(false);
      }
      return;
    }
    socket.send(
      JSON.stringify({
        action: "new_game",
        human_player: humanPlayer,
        preset
      })
    );
  };

  const handleAiPauseToggle = () => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      return;
    }
    socketRef.current.send(
      JSON.stringify({
        action: aiPaused ? "ai_resume" : "ai_pause"
      })
    );
  };

  const handleAiStop = () => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      return;
    }
    socketRef.current.send(
      JSON.stringify({
        action: "ai_stop"
      })
    );
    setAiPaused(false);
  };

  const handleAiStep = () => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      return;
    }
    socketRef.current.send(
      JSON.stringify({
        action: "ai_step"
      })
    );
  };

  const applyEditorAction = (row, col, actionType, button = 0) => {
    const cell = editorBoard[row][col];
    const action = row * editorBoard.length + col;
    const nextBoard = cloneBoard(editorBoard);
    const nextBestActions = new Set(editorBestActions);
    let changed = false;
    let nextLastAction = editorLastAction;
    let nextCurrentPlayer = editorCurrentPlayer;

    if (actionType === "erase" && cell !== 0) {
      nextBoard[row][col] = 0;
      nextBestActions.delete(action);
      if (editorLastAction === action) {
        nextLastAction = -1;
      }
      changed = true;
    } else if (actionType === "best" && cell === 0) {
      if (nextBestActions.has(action)) {
        nextBestActions.delete(action);
      } else {
        nextBestActions.add(action);
      }
      changed = true;
    } else if (actionType === "auto" && cell === 0) {
      nextBoard[row][col] = button === 2 ? 2 : 1;
      nextBestActions.delete(action);
      nextLastAction = action;
      changed = true;
    }

    if (!changed) {
      return;
    }

    setEditorHistory((history) => [
      ...history,
      {
        board: cloneBoard(editorBoard),
        bestActions: Array.from(editorBestActions),
        lastAction: editorLastAction
      }
    ]);
    setEditorBoard(nextBoard);
    setEditorBestActions(nextBestActions);
    setEditorLastAction(nextLastAction);
    nextCurrentPlayer = inferCurrentPlayer(nextBoard);
    setEditorCurrentPlayer(nextCurrentPlayer);
    setSaveStatus("");
    setSaveError("");
  };

  const handleEditorClick = (row, col, button = 0) => {
    applyEditorAction(row, col, editorMode, button);
  };

  const handleEditorUndo = () => {
    if (!editorHistory.length) {
      return;
    }
    const last = editorHistory[editorHistory.length - 1];
    setEditorBoard(last.board);
    setEditorBestActions(new Set(last.bestActions));
    setEditorLastAction(last.lastAction ?? -1);
    setEditorHistory(editorHistory.slice(0, -1));
    setSaveStatus("");
    setSaveError("");
  };

  const handleEditorClear = () => {
    setEditorBoard(makeEmptyBoard(editorBoard.length));
    setEditorBestActions(new Set());
    setEditorHistory([]);
    setEditorLastAction(-1);
    setEditorCurrentPlayer(1);
    setSelectedSampleId("");
    setSaveStatus("");
    setSaveError("");
  };

  const handleUseCurrentBoard = () => {
    setEditorBoard(cloneBoard(board));
    setEditorBestActions(new Set());
    setEditorHistory([]);
    setEditorCurrentPlayer(inferCurrentPlayer(board));
    setEditorLastAction(
      lastMove ? lastMove.x * board.length + lastMove.y : -1
    );
    setSelectedSampleId("");
    setSaveStatus("");
    setSaveError("");
  };

  const resizeEditorBoard = (size) => {
    setEditorBoard(makeEmptyBoard(size));
    setEditorBestActions(new Set());
    setEditorHistory([]);
    setEditorLastAction(-1);
    setEditorCurrentPlayer(1);
    setSelectedSampleId("");
    setSaveStatus("");
    setSaveError("");
  };

  const saveValidationSample = async () => {
    setSaveStatus("");
    setSaveError("");
    const bestActions = Array.from(editorBestActions).sort((a, b) => a - b);
    if (!bestActions.length) {
      setSaveError("Please mark at least one best action.");
      return;
    }
    if (!apiUrl) {
      setSaveError("API URL is empty.");
      return;
    }
    const meta = {};
    if (metaNote.trim()) {
      meta.note = metaNote.trim();
    }
    if (metaTag.trim()) {
      meta.tags = metaTag
        .split(",")
        .map((tag) => tag.trim())
        .filter(Boolean);
    }
    meta.difficulty = metaDifficulty;
    meta.current_player = editorCurrentPlayer;

    const payload = {
      state: {
        board: editorBoard,
        current_player: editorCurrentPlayer,
        last_action: editorLastAction
      },
      best_action: bestActions,
      board_size: editorBoard.length,
      meta
    };

    try {
      setSaving(true);
      window.localStorage.setItem("gomoku_api_url", apiUrl);
      const response = await fetch(`${apiUrl}/api/validation/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) {
        setSaveError(data.detail || "Failed to save.");
        return;
      }
      setSaveStatus(`Saved. id=${data.id}`);
      setSelectedSampleId(data.id);
      fetchValidationList();
    } catch (err) {
      setSaveError("Failed to reach API.");
    } finally {
      setSaving(false);
    }
  };

  const updateValidationSample = async () => {
    if (!selectedSampleId) {
      setSaveError("No sample selected.");
      return;
    }
    setSaveStatus("");
    setSaveError("");
    const bestActions = Array.from(editorBestActions).sort((a, b) => a - b);
    if (!bestActions.length) {
      setSaveError("Please mark at least one best action.");
      return;
    }
    if (!apiUrl) {
      setSaveError("API URL is empty.");
      return;
    }
    const meta = {};
    if (metaNote.trim()) {
      meta.note = metaNote.trim();
    }
    if (metaTag.trim()) {
      meta.tags = metaTag
        .split(",")
        .map((tag) => tag.trim())
        .filter(Boolean);
    }
    meta.difficulty = metaDifficulty;
    meta.current_player = editorCurrentPlayer;

    const payload = {
      id: selectedSampleId,
      state: {
        board: editorBoard,
        current_player: editorCurrentPlayer,
        last_action: editorLastAction
      },
      best_action: bestActions,
      board_size: editorBoard.length,
      meta
    };

    try {
      setSaving(true);
      window.localStorage.setItem("gomoku_api_url", apiUrl);
      const response = await fetch(`${apiUrl}/api/validation/update`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) {
        setSaveError(data.detail || "Failed to update.");
        return;
      }
      setSaveStatus(`Updated. id=${data.id}`);
      fetchValidationList();
    } catch (err) {
      setSaveError("Failed to reach API.");
    } finally {
      setSaving(false);
    }
  };

  const deleteValidationSample = async (sampleId) => {
    if (!apiUrl) {
      setValidationError("API URL is empty.");
      return;
    }
    try {
      setValidationError("");
      const response = await fetch(`${apiUrl}/api/validation/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: sampleId })
      });
      const data = await response.json();
      if (!response.ok) {
        setValidationError(data.detail || "Failed to delete sample.");
        return;
      }
      if (selectedSampleId === sampleId) {
        setSelectedSampleId("");
      }
      fetchValidationList();
    } catch (err) {
      setValidationError("Failed to reach API.");
    }
  };

  const fetchValidationList = async () => {
    if (!apiUrl) {
      setValidationError("API URL is empty.");
      return;
    }
    try {
      setValidationLoading(true);
      setValidationError("");
      const response = await fetch(
        `${apiUrl}/api/validation/list?limit=50`
      );
      const data = await response.json();
      if (!response.ok) {
        setValidationError(data.detail || "Failed to load list.");
        return;
      }
      setValidationItems(data.items || []);
    } catch (err) {
      setValidationError("Failed to reach API.");
    } finally {
      setValidationLoading(false);
    }
  };

  const loadValidationSample = async (sampleId) => {
    if (!apiUrl) {
      setValidationError("API URL is empty.");
      return;
    }
    try {
      setValidationError("");
      const response = await fetch(
        `${apiUrl}/api/validation/get?id=${sampleId}`
      );
      const data = await response.json();
      if (!response.ok) {
        setValidationError(data.detail || "Failed to load sample.");
        return;
      }
      const board = Array.isArray(data.state)
        ? data.state
        : data.state?.board || makeEmptyBoard(DEFAULT_BOARD_SIZE);
      const bestActions = new Set(data.best_action || []);
      setEditorBoard(board);
      setEditorBestActions(bestActions);
      setEditorCurrentPlayer(inferCurrentPlayer(board));
      setEditorLastAction(
        typeof data.state?.last_action === "number"
          ? data.state.last_action
          : -1
      );
      setSelectedSampleId(sampleId);
      setEditorHistory([]);
      setMetaNote(data.meta?.note || "");
      setMetaTag((data.meta?.tags || []).join(", "));
      setMetaDifficulty(
        typeof data.meta?.difficulty === "number" ? data.meta.difficulty : 3
      );
      setSaveStatus("");
      setSaveError("");
    } catch (err) {
      setValidationError("Failed to reach API.");
    }
  };

  const handleEditorKeyDown = (event) => {
    if (event.key.toLowerCase() === "b") {
      event.preventDefault();
      applyEditorAction(editorCursor.row, editorCursor.col, "best", 0);
    }
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
              Challenge the {boardSize}x{boardSize} AI
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
            <label className="mt-4 block text-xs font-semibold uppercase tracking-[0.25em] text-slate-500">
              API URL
            </label>
            <input
              className="mt-2 w-full rounded-xl border border-black/10 bg-white px-3 py-2 text-sm shadow-inner"
              value={apiUrl}
              onChange={(event) => setApiUrl(event.target.value)}
              placeholder={DEFAULT_API}
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
              <span>Mode</span>
              <div className="flex gap-2">
                <button
                  className={`rounded-full px-3 py-1 text-xs font-semibold transition ${
                    matchMode === "human"
                      ? "bg-ink text-white"
                      : "bg-white text-slate-700"
                  }`}
                  onClick={() => setMatchMode("human")}
                >
                  Human vs AI
                </button>
                <button
                  className={`rounded-full px-3 py-1 text-xs font-semibold transition ${
                    matchMode === "ai"
                      ? "bg-ink text-white"
                      : "bg-white text-slate-700"
                  }`}
                  onClick={() => setMatchMode("ai")}
                >
                  AI vs AI
                </button>
              </div>
            </div>
            <div className="mt-4 flex items-center justify-between text-sm text-slate-700">
              <span>Preset board</span>
              <button
                className={`rounded-full px-3 py-1 text-xs font-semibold transition ${
                  usePreset ? "bg-ink text-white" : "bg-white text-slate-700"
                }`}
                onClick={() => setUsePreset((value) => !value)}
              >
                {usePreset ? "Using editor" : "Off"}
              </button>
            </div>
            {matchMode === "human" && (
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
            )}
            {matchMode === "ai" && (
              <div className="mt-4">
                <label className="block text-xs font-semibold uppercase tracking-[0.25em] text-slate-500">
                  AI step delay (ms)
                </label>
                <input
                  className="mt-2 w-full rounded-xl border border-black/10 bg-white px-3 py-2 text-sm shadow-inner"
                  type="number"
                  min="0"
                  value={aiDelayMs}
                  onChange={(event) =>
                    setAiDelayMs(Number(event.target.value || 0))
                  }
                />
              <div className="mt-3 flex gap-2">
                <button
                  className="flex-1 rounded-xl border border-black/10 bg-white px-3 py-2 text-xs font-semibold text-ink transition hover:border-black/30"
                  onClick={handleAiPauseToggle}
                  disabled={!connected || done}
                >
                  {aiPaused ? "Resume" : "Pause"}
                </button>
                <button
                  className="flex-1 rounded-xl border border-black/10 bg-white px-3 py-2 text-xs font-semibold text-ink transition hover:border-black/30"
                  onClick={handleAiStep}
                  disabled={!connected || done || !aiPaused}
                >
                  Step
                </button>
                <button
                  className="flex-1 rounded-xl bg-ink px-3 py-2 text-xs font-semibold text-white transition hover:bg-black"
                  onClick={handleAiStop}
                  disabled={!connected || done}
                >
                    Stop
                  </button>
                </div>
              </div>
            )}
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
              width: `clamp(250px, 76vw, ${Math.max(
                320,
                boardSize * 24
              )}px)`,
              margin: "0 auto"
            }}
          >
            <div
              className="board-surface grid gap-0 rounded-[18px] select-none"
              style={{
                gridTemplateColumns: `repeat(${boardSize}, minmax(0, 1fr))`,
                touchAction: "manipulation",
                backgroundSize: `calc(100% / ${boardSize}) calc(100% / ${boardSize})`
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

          <div className="rounded-2xl border border-black/10 bg-white/80 p-5 text-sm text-slate-700">
            <p className="font-semibold text-slate-900">Validation Set Builder</p>
            <p className="mt-2">
              Draw a position, mark one or more best actions, then save the sample.
            </p>
            {selectedSampleId && (
              <p className="mt-2 text-xs text-slate-500">
                Editing: {selectedSampleId.slice(0, 8)}
              </p>
            )}

            <div className="mt-4 flex flex-wrap gap-2">
              {[
                { key: "auto", label: "Auto" },
                { key: "erase", label: "Erase" },
                { key: "best", label: "Best" }
              ].map((mode) => (
                <button
                  key={mode.key}
                  className={`rounded-full px-3 py-1 text-xs font-semibold transition ${
                    editorMode === mode.key
                      ? "bg-ink text-white"
                      : "bg-white text-slate-700"
                  }`}
                  onClick={() => setEditorMode(mode.key)}
                >
                  {mode.label}
                </button>
              ))}
            </div>

            <div className="mt-3 flex flex-wrap gap-2">
              <button
                className="rounded-xl border border-black/10 bg-white px-3 py-2 text-xs font-semibold text-ink transition hover:border-black/30"
                onClick={handleEditorUndo}
                disabled={!editorHistory.length}
              >
                Undo
              </button>
              <button
                className="rounded-xl border border-black/10 bg-white px-3 py-2 text-xs font-semibold text-ink transition hover:border-black/30"
                onClick={handleEditorClear}
              >
                Clear
              </button>
              <button
                className="rounded-xl border border-black/10 bg-white px-3 py-2 text-xs font-semibold text-ink transition hover:border-black/30"
                onClick={handleUseCurrentBoard}
              >
                Use Current Board
              </button>
            </div>

            <div
              className="mt-4 rounded-[20px] bg-white/70 p-3 shadow-inner"
              style={{
                width: `clamp(220px, 70vw, ${Math.max(
                  300,
                  editorBoard.length * 22
                )}px)`
              }}
            >
              <div
                className="board-surface grid gap-0 rounded-[14px] select-none"
                style={{
                  gridTemplateColumns: `repeat(${editorBoard.length}, minmax(0, 1fr))`,
                  backgroundSize: `calc(100% / ${editorBoard.length}) calc(100% / ${editorBoard.length})`
                }}
                onKeyDown={handleEditorKeyDown}
                tabIndex={0}
              >
                {editorBoard.map((row, rowIndex) =>
                  row.map((cell, colIndex) => {
                    const action = rowIndex * editorBoard.length + colIndex;
                    const isBest = editorBestActions.has(action);
                    return (
                      <button
                        key={`editor-${rowIndex}-${colIndex}`}
                        className="flex aspect-square items-center justify-center bg-transparent transition hover:bg-amber-100/60"
                        onClick={() => handleEditorClick(rowIndex, colIndex, 0)}
                        onContextMenu={(event) => {
                          event.preventDefault();
                          handleEditorClick(rowIndex, colIndex, 2);
                        }}
                        onMouseEnter={() =>
                          setEditorCursor({ row: rowIndex, col: colIndex })
                        }
                        aria-label={`Edit at ${rowIndex}, ${colIndex}`}
                      >
                        {cell !== 0 ? (
                          <span
                            className={`stone h-4 w-4 sm:h-5 sm:w-5 ${
                              cell === 1 ? "stone-black" : "stone-white"
                            }`}
                          />
                        ) : isBest ? (
                          <span className="best-marker" />
                        ) : null}
                      </button>
                    );
                  })
                )}
              </div>
            </div>

            <div className="mt-4 flex items-center justify-between gap-3">
              <div className="text-xs text-slate-600">
                Best actions:{" "}
                {Array.from(editorBestActions)
                  .sort((a, b) => a - b)
                  .join(", ") || "none"}
              </div>
              <div className="flex gap-2">
                <button
                  className="rounded-xl border border-black/10 bg-white px-4 py-2 text-xs font-semibold text-ink transition hover:border-black/30"
                  onClick={saveValidationSample}
                  disabled={saving}
                >
                  Save New
                </button>
                <button
                  className="rounded-xl bg-ink px-4 py-2 text-xs font-semibold text-white transition hover:bg-black"
                  onClick={updateValidationSample}
                  disabled={saving || !selectedSampleId}
                >
                  Update
                </button>
              </div>
            </div>

            <div className="mt-4 grid gap-3 text-xs">
              <div className="flex items-center justify-between gap-3">
                <span className="font-semibold text-slate-700">
                  Current player (inferred)
                </span>
                <span className="rounded-full bg-ink px-3 py-1 text-xs font-semibold text-white">
                  {editorCurrentPlayer === 1 ? "Black" : "White"}
                </span>
              </div>

              <label>
                <span className="font-semibold text-slate-700">Tags</span>
                <input
                  className="mt-1 w-full rounded-xl border border-black/10 bg-white px-3 py-2 text-xs shadow-inner"
                  value={metaTag}
                  onChange={(event) => setMetaTag(event.target.value)}
                  placeholder="comma separated"
                />
              </label>

              <label>
                <span className="font-semibold text-slate-700">Difficulty</span>
                <div className="mt-2 flex flex-wrap gap-2">
                  {[1, 2, 3, 4, 5].map((level) => (
                    <button
                      key={level}
                      className={`rounded-full px-3 py-1 text-xs font-semibold transition ${
                        metaDifficulty === level
                          ? "bg-ink text-white"
                          : "bg-white text-slate-700"
                      }`}
                      onClick={() => setMetaDifficulty(level)}
                    >
                      {level}
                    </button>
                  ))}
                </div>
              </label>

              <label>
                <span className="font-semibold text-slate-700">Note</span>
                <textarea
                  className="mt-1 w-full rounded-xl border border-black/10 bg-white px-3 py-2 text-xs shadow-inner"
                  rows={3}
                  value={metaNote}
                  onChange={(event) => setMetaNote(event.target.value)}
                  placeholder="why is this the best action?"
                />
              </label>
            </div>

            {saveStatus && (
              <p className="mt-3 text-xs font-semibold text-emerald-700">
                {saveStatus}
              </p>
            )}
            {saveError && (
              <p className="mt-3 text-xs font-semibold text-rose-600">
                {saveError}
              </p>
            )}

            <div className="mt-5 border-t border-black/10 pt-4">
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs font-semibold uppercase tracking-[0.25em] text-slate-500">
                  Saved Samples
                </p>
                <div className="flex items-center gap-2">
                  <select
                    className="rounded-full border border-black/10 bg-white px-2 py-1 text-[11px] font-semibold text-ink"
                    value={validationDifficultyFilter}
                    onChange={(event) =>
                      setValidationDifficultyFilter(event.target.value)
                    }
                  >
                    <option value="all">All</option>
                    {[1, 2, 3, 4, 5].map((level) => (
                      <option key={`filter-${level}`} value={`${level}`}>
                        L{level}
                      </option>
                    ))}
                  </select>
                  <button
                    className="rounded-full border border-black/10 bg-white px-3 py-1 text-xs font-semibold text-ink transition hover:border-black/30"
                    onClick={fetchValidationList}
                    disabled={validationLoading}
                  >
                    {validationLoading ? "Loading..." : "Refresh"}
                  </button>
                </div>
              </div>
              {validationError && (
                <p className="mt-2 text-xs font-semibold text-rose-600">
                  {validationError}
                </p>
              )}
              <div className="mt-3 max-h-48 space-y-2 overflow-auto pr-1">
                {validationItems.filter((item) => {
                  if (validationDifficultyFilter === "all") {
                    return true;
                  }
                  const diff = item.meta?.difficulty;
                  return String(diff) === validationDifficultyFilter;
                }).length === 0 && (
                  <p className="text-xs text-slate-500">No samples yet.</p>
                )}
                {validationItems
                  .filter((item) => {
                    if (validationDifficultyFilter === "all") {
                      return true;
                    }
                    const diff = item.meta?.difficulty;
                    return String(diff) === validationDifficultyFilter;
                  })
                  .map((item) => (
                  <div
                    key={item.id}
                    className={`rounded-xl border px-3 py-2 text-xs ${
                      selectedSampleId === item.id
                        ? "border-ink bg-ink/5"
                        : "border-black/10 bg-white"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div>
                    <p className="font-semibold text-slate-900">
                      {item.meta?.tags?.join(", ") || "untagged"}
                    </p>
                    <p className="text-[11px] text-slate-500">
                      {item.id?.slice(0, 8)} •{" "}
                      {item.created_at || "unknown"}
                    </p>
                    <p className="text-[11px] text-slate-500">
                      Difficulty: {item.meta?.difficulty ?? "n/a"}
                    </p>
                      </div>
                      <div className="flex gap-2">
                        <button
                          className="rounded-full bg-ink px-3 py-1 text-xs font-semibold text-white transition hover:bg-black"
                          onClick={() => loadValidationSample(item.id)}
                        >
                          Load
                        </button>
                        <button
                          className="rounded-full border border-black/10 bg-white px-3 py-1 text-xs font-semibold text-ink transition hover:border-black/30"
                          onClick={() => deleteValidationSample(item.id)}
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                    <p className="mt-2 text-[11px] text-slate-600">
                      Best actions: {item.best_action?.join(", ") || "none"}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
