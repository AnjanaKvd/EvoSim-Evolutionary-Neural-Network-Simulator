import { useState, useEffect, useRef, useCallback, memo } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import {
  Play, Pause, SkipForward, RotateCcw, Settings, Dna, Target,
  Wifi, WifiOff, Skull, Globe, Zap, Brain, BarChart2, Crosshair,
  Activity, Radio, AlertTriangle, Cpu, Monitor, Eye
} from "lucide-react";

/* ============================================================
   EvoSim 2.2 - Feature Complete & Optimized
   ============================================================ */

const API_Base = "http://localhost:5000";
const SENSOR_LABELS = ["loc_x", "loc_y", "age", "random", "oscillator", "bdist_x", "bdist_y", "pop_density", "pop_grad_fwd", "genetic_sim_fwd", "last_move_x", "last_move_y", "fwd_blocked", "constant"];
const ACTION_LABELS = ["move_x", "move_y", "move_random", "move_forward", "turn_left", "turn_right", "reverse", "noop/kill"];
const DIRS = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
const formatVal = (n) => typeof n === 'number' ? n.toFixed(2) : n;

// ── Components (Memoized) ───────────────────────────────────

const NeuralDiagram = memo(({ creature }) => {
  if (!creature || !creature.bestConns || !creature.bestConns.length) return (
    <div style={{ padding: 32, textAlign: "center", color: "#2a3b55", fontSize: 12 }}>
      <Cpu size={32} style={{ marginBottom: 8, opacity: 0.5 }} />
      <div>No Neural Data</div>
    </div>
  );

  const conns = creature.bestConns;
  const activeSensors = [...new Set(conns.filter(c => c.sourceType === 0).map(c => c.sourceId))].sort((a, b) => a - b);
  const activeInternal = [...new Set([...conns.filter(c => c.sourceType === 1).map(c => c.sourceId), ...conns.filter(c => c.sinkType === 0).map(c => c.sinkId)])].sort((a, b) => a - b);
  const activeActions = [...new Set(conns.filter(c => c.sinkType === 1).map(c => c.sinkId))].sort((a, b) => a - b);

  const svgW = 300, svgH = 220;
  const yPos = (arr, i) => arr.length <= 1 ? 0.5 : (i / (arr.length - 1)) * 0.8 + 0.1;
  const posMap = {};

  activeSensors.forEach((id, i) => { posMap[`S${id}`] = { x: 0.1, y: yPos(activeSensors, i) }; });
  activeInternal.forEach((id, i) => { posMap[`I${id}`] = { x: 0.5, y: yPos(activeInternal, i) }; });
  activeActions.forEach((id, i) => { posMap[`A${id}`] = { x: 0.9, y: yPos(activeActions, i) }; });

  return (
    <svg viewBox={`0 0 ${svgW} ${svgH}`} style={{ width: "100%", height: "100%", filter: "drop-shadow(0 0 4px rgba(0,0,0,0.5))" }}>
      <defs>
        <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>

      {conns.map((c, i) => {
        const sk = c.sourceType === 0 ? `S${c.sourceId}` : `I${c.sourceId}`;
        const tk = c.sinkType === 1 ? `A${c.sinkId}` : `I${c.sinkId}`;
        const sp = posMap[sk], tp = posMap[tk];
        if (!sp || !tp) return null;

        const x1 = sp.x * svgW, y1 = sp.y * svgH;
        const x2 = tp.x * svgW, y2 = tp.y * svgH;
        const color = c.weight >= 0 ? "#00ff88" : "#ff2244";
        const width = Math.min(3, Math.abs(c.weight) * 1.5) + 0.5;
        const opacity = Math.min(0.8, Math.abs(c.weight) * 0.4 + 0.2);

        return (
          <path key={i}
            d={`M ${x1} ${y1} C ${x1 + 40} ${y1}, ${x2 - 40} ${y2}, ${x2} ${y2}`}
            stroke={color} strokeWidth={width} fill="none" opacity={opacity}
            style={{ animation: "dash 1s linear infinite", strokeDasharray: "4 2" }}
          />
        );
      })}

      {activeSensors.map(id => (
        <g key={`S${id}`} transform={`translate(${posMap[`S${id}`].x * svgW}, ${posMap[`S${id}`].y * svgH})`}>
          <circle r={5} fill="#0d1225" stroke="#00d4ff" strokeWidth={1.5} filter="url(#glow)" />
          <text x={-8} y={3} fill="#4a6080" fontSize={9} textAnchor="end">{SENSOR_LABELS[id]}</text>
        </g>
      ))}
      {activeInternal.map(id => (
        <g key={`I${id}`} transform={`translate(${posMap[`I${id}`].x * svgW}, ${posMap[`I${id}`].y * svgH})`}>
          <circle r={4} fill="#0d1225" stroke="#8899aa" strokeWidth={1.5} />
        </g>
      ))}
      {activeActions.map(id => (
        <g key={`A${id}`} transform={`translate(${posMap[`A${id}`].x * svgW}, ${posMap[`A${id}`].y * svgH})`}>
          <circle r={5} fill="#0d1225" stroke="#ff00ff" strokeWidth={1.5} filter="url(#glow)" />
          <text x={8} y={3} fill="#4a6080" fontSize={9} textAnchor="start">{ACTION_LABELS[id]}</text>
        </g>
      ))}
    </svg>
  );
});

const Slider = memo(({ label, value, min, max, step, onChange, unit }) => {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontSize: 11, color: "#6a7b95" }}>
        <span>{label}</span>
        <span style={{ color: "#00d4ff", fontFamily: "'Share Tech Mono', monospace" }}>{formatVal(value)}{unit}</span>
      </div>
      <div style={{ height: 4, background: "#1a2744", borderRadius: 2, position: "relative" }}>
        <div style={{ position: "absolute", left: 0, top: 0, height: "100%", width: `${pct}%`, background: "#00d4ff", borderRadius: 2 }} />
        <input
          type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))}
          style={{
            position: "absolute", width: "100%", height: 20, top: -8, left: 0, opacity: 0, cursor: "pointer"
          }}
        />
        <div style={{
          position: "absolute", left: `${pct}%`, top: -3, width: 10, height: 10,
          background: "#0a0e1a", border: "2px solid #00d4ff", borderRadius: "50%", transform: "translateX(-50%)",
          boxShadow: "0 0 8px #00d4ff"
        }} />
      </div>
    </div>
  );
});

const StatCard = memo(({ label, value, icon: Icon, color }) => (
  <div style={S.card}>
    <div style={S.cardLabel}><Icon size={12} color={color} /> {label}</div>
    <div style={S.cardValue(color)}>{value}</div>
    <div style={{ position: "absolute", top: 0, right: 0, width: 8, height: 8, borderTop: `1px solid ${color}`, borderRight: `1px solid ${color}` }} />
  </div>
));

const S = {
  root: { display: "flex", flexDirection: "column", height: "100vh", width: "100vw", background: "#0a0e1a", color: "#e0eeff", fontFamily: "'Share Tech Mono', monospace", overflow: "hidden" },
  header: { height: 48, background: "#060910", borderBottom: "1px solid #1a2744", display: "flex", alignItems: "center", padding: "0 16px", justifyContent: "space-between", boxShadow: "0 4px 12px rgba(0,0,0,0.4)", zIndex: 10 },
  logo: { fontFamily: "'Orbitron', sans-serif", fontWeight: 900, fontSize: 18, color: "#00d4ff", textShadow: "0 0 8px rgba(0,212,255,0.4)", letterSpacing: "0.1em", display: "flex", alignItems: "center", gap: 8 },
  leftPanel: { width: 280, background: "rgba(13, 18, 37, 0.95)", borderRight: "1px solid #1a2744", display: "flex", flexDirection: "column", overflowY: "auto", flexShrink: 0 },
  sectionHeader: { fontFamily: "'Orbitron', sans-serif", fontSize: 11, color: "#00d4ff", background: "linear-gradient(90deg, rgba(0,212,255,0.1) 0%, transparent 100%)", padding: "12px 16px", borderBottom: "1px solid #1a2744", borderTop: "1px solid #1a2744", fontWeight: 700, letterSpacing: "0.08em", display: "flex", alignItems: "center", gap: 8, cursor: "pointer", userSelect: "none" },
  ctrlGroup: { padding: "16px", display: "flex", flexDirection: "column", gap: 16 },
  centerPanel: { flex: 1, display: "flex", flexDirection: "column", background: "#0a0e1a", overflow: "hidden", position: "relative" },
  canvasContainer: { flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: 16, overflow: "hidden", position: "relative", backgroundImage: "radial-gradient(circle at center, #111828 0%, #0a0e1a 100%)" },
  statsRow: { display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, padding: "12px 16px", background: "#080c16" },
  chartContainer: { height: 160, padding: "0 16px 12px", background: "#080c16", borderTop: "1px solid #1a2744" },
  rightPanel: { width: 300, background: "#0d1225", borderLeft: "1px solid #1a2744", display: "flex", flexDirection: "column", overflowY: "auto", flexShrink: 0 },
  btn: (active, color = "#00d4ff") => ({ background: active ? `rgba(${parseInt(color.slice(1, 3), 16)}, ${parseInt(color.slice(3, 5), 16)}, ${parseInt(color.slice(5, 7), 16)}, 0.15)` : "transparent", border: `1px solid ${active ? color : "#2a3b55"}`, color: active ? color : "#6a7b95", padding: "8px 16px", borderRadius: 4, cursor: "pointer", fontFamily: "'Orbitron', sans-serif", fontSize: 11, fontWeight: 700, letterSpacing: "0.05em", display: "flex", alignItems: "center", gap: 8, transition: "all 0.2s", boxShadow: active ? `0 0 12px ${color}40` : "none", height: 32 }),
  card: { background: "#0d1829", border: "1px solid #1a2744", borderRadius: 4, padding: "8px 12px", position: "relative", overflow: "hidden", clipPath: "polygon(0 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%)" },
  cardLabel: { fontSize: 10, color: "#4a6080", letterSpacing: "0.05em", textTransform: "uppercase", display: "flex", alignItems: "center", gap: 6 },
  cardValue: (color) => ({ fontSize: 20, fontFamily: "'Orbitron', sans-serif", fontWeight: 700, color, marginTop: 4 }),
};

// ── MAIN APP ────────────────────────────────────────────────

export default function EvoSim() {
  const [cfg, setCfg] = useState({
    worldWidth: 96, worldHeight: 96, population: 400, maxGenerations: 500,
    stepsPerGen: 200, genomeSize: 12, maxInternal: 4, mutationRate: 0.001,
    selectionMode: "west_east", stripWidth: 24, cornerSize: 24, centerRadius: 28,
    killEnabled: false, speed: 3,
  });
  const updateCfg = useCallback((k, v) => setCfg(c => ({ ...c, [k]: v })), []);

  const [visualQuality, setVisualQuality] = useState("low"); // "high" | "low" - Default to low for performance

  // Refs for high-freq data (avoids re-renders)
  const creaturesRef = useRef([]);
  const statsRef = useRef({ gen: 0, survivors: 0, population: 0, diversity: 0, murders: 0, survivalPct: 0 });
  const bestDataRef = useRef({ bestConns: [], bestGenome: [] });
  const runningRef = useRef(false);
  const chartDataRef = useRef([]);

  // UI Sync state (throttled updates)
  const [uiTrigger, setUiTrigger] = useState(0);
  const [runningState, setRunningState] = useState(false);

  useEffect(() => {
    // Fonts & Styles
    const link = document.createElement('link');
    link.href = 'https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap';
    link.rel = 'stylesheet';
    document.head.appendChild(link);
    const style = document.createElement('style');
    style.textContent = `
      @keyframes scanline { 0% { background-position: 0% 0%; } 100% { background-position: 0% 100%; } }
      @keyframes dash { to { stroke-dashoffset: -12; } }
      @keyframes pulse { 0% { opacity: 0.4; } 50% { opacity: 1; } 100% { opacity: 0.4; } }
      ::-webkit-scrollbar { width: 6px; background: #0a0e1a; }
      ::-webkit-scrollbar-thumb { background: #1a2744; borderRadius: 3px; }
      ::-webkit-scrollbar-thumb:hover { background: #00d4ff; }
      input[type=range]::-webkit-slider-thumb { appearance: none; }
    `;
    document.head.appendChild(style);
    return () => { document.head.removeChild(link); document.head.removeChild(style); };
  }, []);

  // API Interaction
  const esRef = useRef(null);

  const connectStream = useCallback(() => {
    if (esRef.current) esRef.current.close();
    const es = new EventSource(`${API_Base}/stream`);
    es.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "generation") {
        creaturesRef.current = data.snapshot || [];
        statsRef.current = {
          gen: data.gen, survivors: data.survivors, population: data.population,
          diversity: data.diversity, murders: data.murders, survivalPct: data.survivalPct
        };
        bestDataRef.current = { bestConns: data.bestConns, bestGenome: data.bestGenome };

        if (chartDataRef.current.length > 100) chartDataRef.current.shift();
        chartDataRef.current.push({
          gen: data.gen, survivors: data.survivors, diversity: data.diversity * 100, murders: data.murders
        });

      } else if (data.type === "done") {
        setRunningState(false);
        runningRef.current = false;
        es.close();
      }
    };
    esRef.current = es;
  }, []);

  const toggleSim = useCallback(async () => {
    if (runningRef.current) {
      await fetch(`${API_Base}/stop`, { method: "POST" });
      setRunningState(false);
      runningRef.current = false;
      if (esRef.current) esRef.current.close();
    } else {
      setRunningState(true);
      runningRef.current = true;
      // Map cfg to backend expected keys if needed, but backend seems to handle camelCase -> snake_case somewhat?
      // Actually backend keys are manual: "stepsPerGen" -> "steps_per_gen"
      await fetch(`${API_Base}/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cfg)
      });
      chartDataRef.current = [];
      connectStream();
    }
  }, [cfg, connectStream]);

  const resetSim = useCallback(async () => {
    await fetch(`${API_Base}/stop`, { method: "POST" });
    creaturesRef.current = [];
    statsRef.current = { gen: 0, survivors: 0, population: 0, diversity: 0, murders: 0, survivalPct: 0 };
    bestDataRef.current = { bestConns: [], bestGenome: [] };
    chartDataRef.current = [];
    setUiTrigger(t => t + 1);
    setRunningState(false);
    runningRef.current = false;
    if (esRef.current) esRef.current.close();
  }, []);

  // UI Throttle Loop (10fps)
  useEffect(() => {
    const interval = setInterval(() => {
      if (runningRef.current) {
        setUiTrigger(t => t + 1);
      }
    }, 100);
    return () => clearInterval(interval);
  }, []);

  // Canvas Render Loop (60fps decoupled)
  const canvasRef = useRef(null);

  // Need a ref to access current Visual Quality inside the loop without re-binding
  const vqRef = useRef(visualQuality);
  useEffect(() => { vqRef.current = visualQuality; }, [visualQuality]);

  useEffect(() => {
    const cvs = canvasRef.current;
    if (!cvs) return;
    const ctx = cvs.getContext("2d", { alpha: false });

    let animationFrameId;

    const render = () => {
      const container = cvs.parentElement;
      if (container) {
        const size = Math.min(container.clientWidth, container.clientHeight);
        if (cvs.width !== size) { cvs.width = size; cvs.height = size; }

        const W = cfg.worldWidth, H = cfg.worldHeight; // Dynamic height
        const cell = size / Math.max(W, H);

        const quality = vqRef.current; // "high" or "low"

        // Clear Logic
        if (quality === "high") {
          // Trails
          ctx.fillStyle = "rgba(10, 14, 26, 0.2)";
          ctx.fillRect(0, 0, size, size);
        } else {
          // Instant Clear (Fast)
          ctx.fillStyle = "#0a0e1a";
          ctx.fillRect(0, 0, size, size);
        }

        // Grid (draw less frequently? No, static is cheap)
        ctx.strokeStyle = quality === "high" ? "rgba(42, 59, 85, 0.15)" : "rgba(42, 59, 85, 0.05)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i <= W; i += 8) { ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, size); }
        for (let i = 0; i <= H; i += 8) { ctx.moveTo(0, i * cell); ctx.lineTo(size, i * cell); }
        ctx.stroke();

        // Selection Zone
        ctx.fillStyle = "rgba(0, 255, 136, 0.08)";
        const mode = cfg.selectionMode;
        if (mode === "east") ctx.fillRect(size / 2, 0, size / 2, size);
        else if (mode === "west") ctx.fillRect(0, 0, size / 2, size);
        else if (mode === "west_east") { // "Both Sides" (user calls it)
          const strip = cfg.stripWidth * cell;
          ctx.fillRect(0, 0, strip, size); // West
          ctx.fillRect(size - strip, 0, strip, size); // East
        }
        else if (mode === "center") {
          ctx.beginPath(); ctx.arc(size / 2, size / 2, cfg.centerRadius * cell, 0, Math.PI * 2);
          ctx.fill();
        }
        else if (mode === "corners") {
          const cs = cfg.cornerSize * cell;
          ctx.fillRect(0, 0, cs, cs); ctx.fillRect(size - cs, 0, cs, cs);
          ctx.fillRect(0, size - cs, cs, cs); ctx.fillRect(size - cs, size - cs, cs, cs);
        } else if (mode === "radioactive") {
          // Red walls?
          ctx.fillStyle = "rgba(255, 68, 68, 0.1)";
          ctx.fillRect(0, 0, 10, size); ctx.fillRect(size - 10, 0, 10, size);
        }

        // Creatures
        const creatures = creaturesRef.current;
        for (let i = 0; i < creatures.length; i++) {
          const c = creatures[i];
          const cx = (c.x + 0.5) * cell;
          const cy = (c.y + 0.5) * cell;
          const dx = DIRS[c.d][0];
          const dy = DIRS[c.d][1];
          const angle = Math.atan2(dy, dx);

          ctx.save();
          ctx.translate(cx, cy);
          ctx.rotate(angle);

          ctx.fillStyle = `rgb(${c.r},${c.g},${c.b})`;

          if (quality === "high") {
            // Fancy Triangle
            const scale = cell * 0.7;
            ctx.beginPath();
            ctx.moveTo(scale, 0);
            ctx.lineTo(-scale, -scale * 0.6);
            ctx.lineTo(-scale, scale * 0.6);
            ctx.fill();
          } else {
            // Simple Dot/Triangle
            const scale = cell * 0.6;
            ctx.beginPath();
            ctx.moveTo(scale, 0);
            ctx.lineTo(-scale, -scale * 0.5);
            ctx.lineTo(-scale, scale * 0.5);
            ctx.fill();
          }

          ctx.restore();
        }
      }
      animationFrameId = requestAnimationFrame(render);
    };
    render();
    return () => cancelAnimationFrame(animationFrameId);
  }, [cfg.worldWidth, cfg.worldHeight, cfg.selectionMode, cfg.centerRadius, cfg.cornerSize, cfg.stripWidth]);

  const stats = statsRef.current;
  const bestData = bestDataRef.current;

  return (
    <div style={S.root}>
      {/* Header */}
      <header style={S.header}>
        <div style={S.logo}>
          <Dna size={24} color="#00d4ff" /> EVOSIM <span style={{ fontSize: 10, opacity: 0.6 }}>v2.2 (OPTIMIZED)</span>
        </div>
        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          {/* Visual Quality Toggle */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, border: "1px solid #1a2744", borderRadius: 4, padding: 2 }}>
            <button onClick={() => setVisualQuality("low")} style={{ ...S.btn(visualQuality === "low"), height: 24, fontSize: 10 }}><Monitor size={12} /> LOW</button>
            <button onClick={() => setVisualQuality("high")} style={{ ...S.btn(visualQuality === "high"), height: 24, fontSize: 10 }}><Eye size={12} /> HIGH</button>
          </div>

          <div style={{ color: runningState ? "#00ff88" : "#6a7b95", fontSize: 12, display: "flex", alignItems: "center", gap: 6, fontFamily: "'Orbitron'" }}>
            <Activity size={14} /> {runningState ? "RUNNING" : "STANDBY"}
          </div>
          <div style={{ fontFamily: "'Orbitron'", fontWeight: 700, fontSize: 14, color: "#fff" }}>
            GEN {stats.gen}
          </div>
        </div>
      </header>

      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* Left Settings */}
        <aside style={S.leftPanel}>
          <div style={S.sectionHeader}><Globe size={14} /> WORLD</div>
          <div style={S.ctrlGroup}>
            <Slider label="Width" min={32} max={128} step={8} value={cfg.worldWidth} onChange={v => updateCfg("worldWidth", v)} unit="px" />
            <Slider label="Height" min={32} max={128} step={8} value={cfg.worldHeight} onChange={v => updateCfg("worldHeight", v)} unit="px" />
            <Slider label="Population" min={50} max={1000} step={50} value={cfg.population} onChange={v => updateCfg("population", v)} />
            <Slider label="Max Gen" min={100} max={2000} step={100} value={cfg.maxGenerations} onChange={v => updateCfg("maxGenerations", v)} />
            <Slider label="Steps/Gen" min={50} max={500} step={50} value={cfg.stepsPerGen} onChange={v => updateCfg("stepsPerGen", v)} />
          </div>

          <div style={S.sectionHeader}><Dna size={14} /> GENOME</div>
          <div style={S.ctrlGroup}>
            <Slider label="Genome Size" min={2} max={32} step={1} value={cfg.genomeSize} onChange={v => updateCfg("genomeSize", v)} />
            <Slider label="Internal Neurons" min={0} max={16} step={1} value={cfg.maxInternal} onChange={v => updateCfg("maxInternal", v)} />
            <Slider label="Mutation Rate" min={0.0001} max={0.01} step={0.0001} value={cfg.mutationRate} onChange={v => updateCfg("mutationRate", v)} />
          </div>

          <div style={S.sectionHeader}><Target size={14} /> SELECTION</div>
          <div style={S.ctrlGroup}>
            <select style={{ background: "#0a0e1a", border: "1px solid #1a2744", color: "#00d4ff", padding: 8, fontFamily: "'Share Tech Mono'", marginBottom: 12 }}
              value={cfg.selectionMode} onChange={e => updateCfg("selectionMode", e.target.value)}>
              <option value="west_east">Both Sides</option>
              <option value="east">East World</option>
              <option value="west">West World</option>
              <option value="center">Center Zone</option>
              <option value="corners">Remote Corners</option>
              <option value="radioactive">Radioactive Walls</option>
            </select>

            {cfg.selectionMode === "center" && (
              <Slider label="Radius" min={5} max={40} step={1} value={cfg.centerRadius} onChange={v => updateCfg("centerRadius", v)} />
            )}
            {(cfg.selectionMode === "west_east" || cfg.selectionMode === "east" || cfg.selectionMode === "west") && (
              <Slider label="Strip Width" min={8} max={48} step={4} value={cfg.stripWidth} onChange={v => updateCfg("stripWidth", v)} />
            )}
            {cfg.selectionMode === "corners" && (
              <Slider label="Corner Size" min={8} max={48} step={4} value={cfg.cornerSize} onChange={v => updateCfg("cornerSize", v)} />
            )}

            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 8 }}>
              <span style={{ fontSize: 11, color: "#6a7b95" }}>Kill Enabled</span>
              <input type="checkbox" checked={cfg.killEnabled} onChange={e => updateCfg("killEnabled", e.target.checked)} style={{ accentColor: "#ff2244" }} />
            </div>
          </div>

          <div style={S.sectionHeader}><Play size={14} /> PLAYBACK</div>
          <div style={S.ctrlGroup}>
            <Slider label="Speed" min={1} max={5} step={1} value={cfg.speed} onChange={v => updateCfg("speed", v)} />
            <div style={{ textAlign: "center", fontSize: 11, color: "#6a7b95", marginTop: -8, letterSpacing: 2 }}>
              {["SLOW", "NORMAL", "FAST", "TURBO", "MAX"][cfg.speed - 1]}
            </div>
          </div>

          <div style={{ marginTop: "auto", padding: 16 }}>
            <button style={{ ...S.btn(true, runningState ? "#aa0000" : "#00ff88"), width: "100%", height: 48, fontSize: 14 }} onClick={toggleSim}>
              {runningState ? <Pause size={18} /> : <Play size={18} />} {runningState ? "ABORT" : "START"}
            </button>
            <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
              <button style={{ ...S.btn(false), flex: 1 }} onClick={() => resetSim()}><RotateCcw size={14} /> RESET</button>
            </div>
          </div>
        </aside>

        {/* Center Canvas */}
        <main style={S.centerPanel}>
          <div style={S.canvasContainer}>
            <div style={{ position: "absolute", top: 20, left: 20, width: 20, height: 20, borderTop: "2px solid #00d4ff", borderLeft: "2px solid #00d4ff" }} />
            <div style={{ position: "absolute", top: 20, right: 20, width: 20, height: 20, borderTop: "2px solid #00d4ff", borderRight: "2px solid #00d4ff" }} />
            <div style={{ position: "absolute", bottom: 20, left: 20, width: 20, height: 20, borderBottom: "2px solid #00d4ff", borderLeft: "2px solid #00d4ff" }} />
            <div style={{ position: "absolute", bottom: 20, right: 20, width: 20, height: 20, borderBottom: "2px solid #00d4ff", borderRight: "2px solid #00d4ff" }} />
            <div style={{ position: "absolute", inset: 0, background: "linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06))", backgroundSize: "100% 4px, 6px 100%", pointerEvents: "none", zIndex: 10 }} />
            <canvas ref={canvasRef} style={{ maxWidth: "90%", maxHeight: "90%", background: "#000", border: "1px solid #1a2744", boxShadow: "0 0 30px rgba(0,0,0,0.5)" }} />
            {!runningState && stats.gen === 0 && (
              <div style={{ position: "absolute", color: "#00d4ff", fontFamily: "'Orbitron'", letterSpacing: 4, background: "rgba(0,0,0,0.8)", padding: "16px 32px", border: "1px solid #00d4ff" }}>
                SYSTEM READY
              </div>
            )}
          </div>
          <div style={S.statsRow}>
            <StatCard label="SURVIVAL RATE" value={stats.survivalPct + "%"} icon={Activity} color={stats.survivalPct > 80 ? "#00ff88" : "#ffaa00"} />
            <StatCard label="POPULATION" value={stats.population} icon={Globe} color="#00d4ff" />
            <StatCard label="DIVERSITY" value={stats.diversity} icon={Dna} color="#bb00ff" />
            <StatCard label="FATALITIES" value={stats.murders} icon={Skull} color="#ff2244" />
          </div>
          <div style={S.chartContainer}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartDataRef.current}>
                <XAxis dataKey="gen" tick={{ fontSize: 10, fill: "#4a6080" }} axisLine={false} tickLine={false} />
                <YAxis dataKey="survivors" tick={{ fontSize: 10, fill: "#4a6080" }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: "#0d1225", border: "1px solid #1a2744", fontSize: 12 }} />
                <Line type="monotone" dataKey="survivors" stroke="#00ff88" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="diversity" stroke="#bb00ff" strokeWidth={1} dot={false} isAnimationActive={false} strokeDasharray="4 4" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </main>

        {/* Right Neural */}
        <aside style={S.rightPanel}>
          <div style={S.sectionHeader}><Brain size={14} /> NEURAL LINK</div>
          <div style={{ height: 240, borderBottom: "1px solid #1a2744", background: "#080c12" }}>
            <NeuralDiagram creature={bestData} />
          </div>
          <div style={S.sectionHeader}><Cpu size={14} /> CONNECTION LOG</div>
          <div style={{ flex: 1, overflowY: "auto", fontSize: 10, padding: 8 }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <tbody>
                {bestData.bestConns && bestData.bestConns.map((c, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid #1a2744", color: c.weight > 0 ? "#00ff88" : "#ff2244" }}>
                    <td style={{ padding: 4 }}>{c.sourceType === 0 ? SENSOR_LABELS[c.sourceId] : "I" + c.sourceId}</td>
                    <td style={{ padding: 4 }}>→</td>
                    <td style={{ padding: 4 }}>{c.sinkType === 1 ? ACTION_LABELS[c.sinkId] : "I" + c.sinkId}</td>
                    <td style={{ padding: 4, textAlign: "right" }}>{formatVal(c.weight)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </aside>
      </div>
    </div>
  );
}
