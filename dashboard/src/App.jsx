import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";

/* ============================================================
   SIMULATION ENGINE (full JS port of the Python EvoSim)
   ============================================================ */

// ── Genome ──────────────────────────────────────────────────
function randomGenome(size, rng) {
  return Array.from({ length: size }, () => ((rng() * 0xFFFFFFFF) | 0));
}

function decodeGene(gene, numSensors, numActions, maxInternal) {
  const u = gene >>> 0;
  const sourceType = (u >>> 31) & 1;
  let sourceId = (u >>> 24) & 0x7F;
  const sinkType = (u >>> 23) & 1;
  let sinkId = (u >>> 16) & 0x7F;
  let rawW = u & 0xFFFF;
  if (rawW >= 0x8000) rawW -= 0x10000;
  const weight = rawW / 8000;
  sourceId = sourceType === 0 ? sourceId % numSensors : sourceId % Math.max(1, maxInternal);
  sinkId = sinkType === 1 ? sinkId % numActions : sinkId % Math.max(1, maxInternal);
  return { sourceType, sourceId, sinkType, sinkId, weight };
}

function mutateGenome(genome, rate, rng) {
  return genome.map(gene => {
    let g = gene >>> 0;
    for (let b = 0; b < 32; b++) if (rng() < rate) g ^= (1 << b);
    g = g >>> 0;
    return g >= 0x80000000 ? g - 0x100000000 : g;
  });
}

function crossover(ga, gb, rng) {
  const split = Math.floor(rng() * (ga.length + 1));
  return [...ga.slice(0, split), ...gb.slice(split)];
}

function genomeSimilarity(ga, gb) {
  let match = 0, total = 0;
  for (let i = 0; i < Math.min(ga.length, gb.length); i++) {
    const xor = (ga[i] ^ gb[i]) >>> 0;
    match += 32 - popcount(xor); total += 32;
  }
  return total ? match / total : 1;
}

function popcount(n) {
  n = n >>> 0;
  n -= (n >> 1) & 0x55555555;
  n = (n & 0x33333333) + ((n >> 2) & 0x33333333);
  n = (n + (n >> 4)) & 0x0f0f0f0f;
  return (n * 0x01010101) >>> 24;
}

function genomeToColor(genome) {
  let h = 0;
  for (const g of genome) h ^= (g >>> 0) & 0xFFFFFF;
  let r = (h >> 16) & 0xFF, g = (h >> 8) & 0xFF, b = h & 0xFF;
  return [Math.max(60, r), Math.max(60, g), Math.max(60, b)];
}

// ── Neural Network ────────────────────────────────────────────
const NUM_SENSORS = 14, NUM_ACTIONS = 8;
const SENSOR_LABELS = ["loc_x", "loc_y", "age", "random", "oscillator", "bdist_x", "bdist_y", "pop_density", "pop_grad_fwd", "genetic_sim_fwd", "last_move_x", "last_move_y", "fwd_blocked", "constant"];
const ACTION_LABELS = ["move_x", "move_y", "move_random", "move_forward", "turn_left", "turn_right", "reverse", "noop/kill"];
const DIRS = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];

class NeuralNet {
  constructor(genome, maxInternal) {
    this.maxInternal = maxInternal;
    this.conns = this._parse(genome);
    this.conns = this._prune(this.conns);
  }
  _parse(genome) {
    return genome.map(g => decodeGene(g, NUM_SENSORS, NUM_ACTIONS, this.maxInternal));
  }
  _prune(conns) {
    const useful = new Set();
    let changed = true;
    while (changed) {
      changed = false;
      for (const c of conns) {
        if (c.sinkType === 1) {
          if (c.sourceType === 1 && !useful.has(c.sourceId)) { useful.add(c.sourceId); changed = true; }
        } else if (c.sinkType === 0 && useful.has(c.sinkId)) {
          if (c.sourceType === 1 && !useful.has(c.sourceId)) { useful.add(c.sourceId); changed = true; }
        }
      }
    }
    return conns.filter(c => c.sinkType === 1 || useful.has(c.sinkId));
  }
  forward(sensors) {
    const internal = new Float32Array(this.maxInternal);
    for (let pass = 0; pass < 2; pass++) {
      const next = new Float32Array(this.maxInternal);
      for (const c of this.conns) {
        if (c.sinkType === 0) {
          const src = c.sourceType === 0 ? sensors[c.sourceId] : internal[c.sourceId];
          next[c.sinkId] += src * c.weight;
        }
      }
      for (let i = 0; i < this.maxInternal; i++) internal[i] = Math.tanh(next[i]);
    }
    const actions = new Float32Array(NUM_ACTIONS);
    for (const c of this.conns) {
      if (c.sinkType === 1) {
        const src = c.sourceType === 0 ? sensors[c.sourceId] : internal[c.sourceId];
        actions[c.sinkId] += src * c.weight;
      }
    }
    for (let i = 0; i < NUM_ACTIONS; i++) actions[i] = Math.tanh(actions[i]);
    return actions;
  }
}

// ── Creature ───────────────────────────────────────────────────
class Creature {
  constructor(x, y, genome, maxInternal) {
    this.x = x; this.y = y;
    this.genome = genome;
    this.brain = new NeuralNet(genome, maxInternal);
    this.alive = true;
    this.age = 0;
    this.lastDir = 0;
    this.oscPhase = 0;
    this.radiationDose = 0;
    this.color = genomeToColor(genome);
  }
}

// ── World ───────────────────────────────────────────────────────
class World {
  constructor(W, H) {
    this.W = W; this.H = H;
    this.grid = new Array(W * H).fill(null);
    this.creatures = [];
    this.murders = 0;
    this.rng = seededRng(Date.now());
  }
  idx(x, y) { return y * this.W + x; }
  inBounds(x, y) { return x >= 0 && x < this.W && y >= 0 && y < this.H; }
  get(x, y) { return this.inBounds(x, y) ? this.grid[this.idx(x, y)] : null; }
  set(x, y, c) { if (this.inBounds(x, y)) this.grid[this.idx(x, y)] = c; }
  clear() { this.grid.fill(null); this.creatures = []; this.murders = 0; }

  populate(creatures) {
    this.clear();
    const positions = shuffle(Array.from({ length: this.W * this.H }, (_, i) => i), this.rng);
    for (let i = 0; i < Math.min(creatures.length, positions.length); i++) {
      const pos = positions[i];
      const x = pos % this.W, y = Math.floor(pos / this.W);
      creatures[i].x = x; creatures[i].y = y;
      this.grid[pos] = creatures[i];
    }
    this.creatures = creatures;
  }

  move(creature, dx, dy) {
    const nx = Math.max(0, Math.min(this.W - 1, creature.x + dx));
    const ny = Math.max(0, Math.min(this.H - 1, creature.y + dy));
    if (nx === creature.x && ny === creature.y) return false;
    if (this.get(nx, ny) !== null) return false;
    this.set(creature.x, creature.y, null);
    creature.x = nx; creature.y = ny;
    this.set(nx, ny, creature);
    return true;
  }

  killAt(x, y) {
    const v = this.get(x, y);
    if (v && v.alive) { v.alive = false; this.set(x, y, null); this.murders++; }
  }

  localDensity(cx, cy, r = 2) {
    let count = 0;
    for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
      if (dx === 0 && dy === 0) continue;
      if (this.inBounds(cx + dx, cy + dy) && this.get(cx + dx, cy + dy)) count++;
    }
    return count;
  }

  stepCreature(creature, simStep, stepsPerGen, killEnabled) {
    if (!creature.alive) return;
    creature.age++;
    creature.oscPhase += (2 * Math.PI) / 30;

    // Sense
    const s = new Float32Array(NUM_SENSORS);
    s[0] = creature.x / (this.W - 1);
    s[1] = creature.y / (this.H - 1);
    s[2] = creature.age / Math.max(1, stepsPerGen - 1);
    s[3] = this.rng();
    s[4] = (Math.sin(creature.oscPhase) + 1) / 2;
    s[5] = 1 - Math.min(creature.x, this.W - 1 - creature.x) / (this.W / 2);
    s[6] = 1 - Math.min(creature.y, this.H - 1 - creature.y) / (this.H / 2);
    s[7] = Math.min(1, this.localDensity(creature.x, creature.y) / 8);
    const [ddx, ddy] = DIRS[creature.lastDir];
    let fwd = 0, bwd = 0;
    for (let step = 1; step <= 5; step++) {
      if (this.inBounds(creature.x + ddx * step, creature.y + ddy * step) && this.get(creature.x + ddx * step, creature.y + ddy * step)) fwd++;
      if (this.inBounds(creature.x - ddx * step, creature.y - ddy * step) && this.get(creature.x - ddx * step, creature.y - ddy * step)) bwd++;
    }
    s[8] = (fwd - bwd) / 5;
    const ahead = this.get(creature.x + ddx, creature.y + ddy);
    s[9] = ahead ? genomeSimilarity(creature.genome, ahead.genome) : 0;
    s[10] = (DIRS[creature.lastDir][0] + 1) / 2;
    s[11] = (DIRS[creature.lastDir][1] + 1) / 2;
    s[12] = ahead ? 1 : 0;
    s[13] = 1;

    // Think
    const actions = creature.brain.forward(s);
    let bestAct = -1, bestVal = 0.1;
    for (let i = 0; i < NUM_ACTIONS; i++) {
      if (Math.abs(actions[i]) > bestVal) { bestVal = Math.abs(actions[i]); bestAct = i; }
    }
    if (bestAct < 0) return;

    // Act
    const str = actions[bestAct];
    if (bestAct === 0) { const dx2 = str > 0 ? 1 : -1; if (this.move(creature, dx2, 0)) { creature.lastDir = str > 0 ? 0 : 4; } }
    else if (bestAct === 1) { const dy2 = str > 0 ? 1 : -1; if (this.move(creature, 0, dy2)) { creature.lastDir = str > 0 ? 2 : 6; } }
    else if (bestAct === 2) { const d = Math.floor(this.rng() * 8); if (this.move(creature, DIRS[d][0], DIRS[d][1])) creature.lastDir = d; }
    else if (bestAct === 3) { this.move(creature, DIRS[creature.lastDir][0], DIRS[creature.lastDir][1]); }
    else if (bestAct === 4) { creature.lastDir = (creature.lastDir + 7) % 8; this.move(creature, DIRS[creature.lastDir][0], DIRS[creature.lastDir][1]); }
    else if (bestAct === 5) { creature.lastDir = (creature.lastDir + 1) % 8; this.move(creature, DIRS[creature.lastDir][0], DIRS[creature.lastDir][1]); }
    else if (bestAct === 6) { creature.lastDir = (creature.lastDir + 4) % 8; this.move(creature, DIRS[creature.lastDir][0], DIRS[creature.lastDir][1]); }
    else if (bestAct === 7 && killEnabled) { this.killAt(creature.x + DIRS[creature.lastDir][0], creature.y + DIRS[creature.lastDir][1]); }
  }
}

// ── Helpers ─────────────────────────────────────────────────
function seededRng(seed) {
  let s = seed >>> 0;
  return () => { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return (s >>> 0) / 0xFFFFFFFF; };
}

function shuffle(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function select(creatures, mode, W, H, cfg) {
  return creatures.filter(c => {
    if (!c.alive) return false;
    if (mode === "east") return c.x >= W / 2;
    if (mode === "west") return c.x < W / 2;
    if (mode === "west_east") return c.x < cfg.stripWidth || c.x >= W - cfg.stripWidth;
    if (mode === "corners") return (c.x < cfg.cornerSize || c.x >= W - cfg.cornerSize) && (c.y < cfg.cornerSize || c.y >= H - cfg.cornerSize);
    if (mode === "center") { const cx = W / 2, cy = H / 2; return (c.x - cx) ** 2 + (c.y - cy) ** 2 <= cfg.centerRadius ** 2; }
    if (mode === "radioactive") return c.alive;
    return true;
  });
}

function reproduce(survivors, pop, genomeSize, mutationRate, rng) {
  if (!survivors.length) return Array.from({ length: pop }, () => randomGenome(genomeSize, rng));
  const next = [];
  while (next.length < pop) {
    const pa = survivors[Math.floor(rng() * survivors.length)];
    const pb = survivors[Math.floor(rng() * survivors.length)];
    let child = crossover(pa.genome, pb.genome, rng);
    child = mutateGenome(child, mutationRate, rng);
    next.push(child);
  }
  return next;
}

function calcDiversity(survivors) {
  if (survivors.length < 2) return 0;
  const sample = survivors.slice(0, Math.min(30, survivors.length));
  let total = 0, count = 0;
  for (let i = 0; i < sample.length; i++)
    for (let j = i + 1; j < sample.length; j++) {
      total += 1 - genomeSimilarity(sample[i].genome, sample[j].genome);
      count++;
    }
  return count ? total / count : 0;
}

/* ============================================================
   REACT COMPONENTS
   ============================================================ */

// (WorldCanvas removed — rendering is done directly via canvasWorldRef)

// Neural Network SVG Diagram
function NeuralDiagram({ creature }) {
  if (!creature) return <div style={{ color: "#555", textAlign: "center", paddingTop: 60, fontSize: 12 }}>No creature selected</div>;
  const conns = creature.brain.conns;

  const activeSensors = [...new Set(conns.filter(c => c.sourceType === 0).map(c => c.sourceId))].sort((a, b) => a - b);
  const activeInternal = [...new Set([
    ...conns.filter(c => c.sourceType === 1).map(c => c.sourceId),
    ...conns.filter(c => c.sinkType === 0).map(c => c.sinkId)
  ])].sort((a, b) => a - b);
  const activeActions = [...new Set(conns.filter(c => c.sinkType === 1).map(c => c.sinkId))].sort((a, b) => a - b);

  const svgW = 320, svgH = 240;
  const posMap = {};
  const yPos = (arr, idx) => arr.length <= 1 ? 0.5 : idx / (arr.length - 1);

  activeSensors.forEach((id, i) => { posMap[`S${id}`] = { x: 0.08, y: 0.05 + 0.9 * yPos(activeSensors, i) }; });
  activeInternal.forEach((id, i) => { posMap[`I${id}`] = { x: 0.5, y: 0.05 + 0.9 * yPos(activeInternal, i) }; });
  activeActions.forEach((id, i) => { posMap[`A${id}`] = { x: 0.92, y: 0.05 + 0.9 * yPos(activeActions, i) }; });

  const edges = conns.map((c, i) => {
    const sk = c.sourceType === 0 ? `S${c.sourceId}` : `I${c.sourceId}`;
    const tk = c.sinkType === 1 ? `A${c.sinkId}` : `I${c.sinkId}`;
    const sp = posMap[sk], tp = posMap[tk];
    if (!sp || !tp) return null;
    const x1 = sp.x * svgW, y1 = sp.y * svgH, x2 = tp.x * svgW, y2 = tp.y * svgH;
    const mx = (x1 + x2) / 2;
    const color = c.weight >= 0 ? "#22ee77" : "#ff4455";
    const opacity = 0.3 + Math.min(0.7, Math.abs(c.weight) * 0.5);
    const sw = 0.5 + Math.min(2.5, Math.abs(c.weight));
    return <path key={i} d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`}
      stroke={color} strokeWidth={sw} fill="none" opacity={opacity} />;
  }).filter(Boolean);

  const NodeCircle = ({ id, col, color, labelMap }) => {
    const p = posMap[id];
    if (!p) return null;
    const isLeft = col === "S";
    const label = labelMap ? labelMap[parseInt(id.slice(1))] : id;
    return (
      <g>
        <circle cx={p.x * svgW} cy={p.y * svgH} r={6} fill={color} opacity={0.9} />
        <text x={p.x * svgW + (isLeft ? -10 : col === "A" ? 10 : 0)}
          y={p.y * svgH + 4}
          fill="#ccc" fontSize={7}
          textAnchor={isLeft ? "end" : col === "A" ? "start" : "middle"}>{label}</text>
      </g>
    );
  };

  return (
    <svg viewBox={`0 0 ${svgW} ${svgH}`} style={{ width: "100%", height: "100%" }}>
      <text x={svgW * 0.08} y={12} fill="#4488ff" fontSize={8} textAnchor="middle">SENSORS</text>
      <text x={svgW * 0.5} y={12} fill="#aaaaaa" fontSize={8} textAnchor="middle">INTERNAL</text>
      <text x={svgW * 0.92} y={12} fill="#ff88aa" fontSize={8} textAnchor="middle">ACTIONS</text>
      {edges}
      {activeSensors.map(id => <NodeCircle key={`S${id}`} id={`S${id}`} col="S" color="#4488ff" labelMap={SENSOR_LABELS} />)}
      {activeInternal.map(id => <NodeCircle key={`I${id}`} id={`I${id}`} col="I" color="#888888" />)}
      {activeActions.map(id => <NodeCircle key={`A${id}`} id={`A${id}`} col="A" color="#ff88aa" labelMap={ACTION_LABELS} />)}
      <text x={svgW - 4} y={svgH - 4} fill="#333" fontSize={6} textAnchor="end">
        {conns.length} active connections
      </text>
    </svg>
  );
}

// Slider control
function Ctrl({ label, value, min, max, step, onChange, unit = "" }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 13, color: "#a8bece", letterSpacing: "0.04em", textTransform: "uppercase" }}>{label}</span>
        <span style={{ fontSize: 13, color: "#00e87a", fontFamily: "monospace", fontWeight: "bold" }}>{value}{unit}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: "100%", accentColor: "#00e87a", height: 4, cursor: "pointer" }} />
    </div>
  );
}

function Select({ label, value, options, onChange }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ fontSize: 13, color: "#a8bece", letterSpacing: "0.04em", textTransform: "uppercase", marginBottom: 5 }}>{label}</div>
      <select value={value} onChange={e => onChange(e.target.value)}
        style={{ width: "100%", background: "#0d1a28", color: "#00e87a", border: "1px solid #2a4060", borderRadius: 4, padding: "5px 8px", fontSize: 13, fontFamily: "monospace" }}>
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  );
}

function Toggle({ label, value, onChange }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
      <span style={{ fontSize: 13, color: "#a8bece", letterSpacing: "0.04em", textTransform: "uppercase" }}>{label}</span>
      <div onClick={() => onChange(!value)} style={{
        width: 40, height: 22, borderRadius: 11,
        background: value ? "#00cc66" : "#2a4060", cursor: "pointer", position: "relative", transition: "background 0.2s"
      }}>
        <div style={{
          position: "absolute", top: 3, left: value ? 20 : 3, width: 16, height: 16,
          borderRadius: "50%", background: value ? "#fff" : "#7a9ab8", transition: "left 0.2s"
        }} />
      </div>
    </div>
  );
}

/* ============================================================
   MAIN DASHBOARD
   ============================================================ */
export default function EvoSim() {
  // ── Config ────────────────────────────────────────────────
  const [cfg, setCfg] = useState({
    worldWidth: 96, worldHeight: 96,
    population: 400, maxGenerations: 500,
    stepsPerGen: 200, genomeSize: 12,
    maxInternal: 4, mutationRate: 0.001,
    selectionMode: "east",
    stripWidth: 24, cornerSize: 24,
    centerRadius: 18,
    killEnabled: false,
    speed: 3,
  });

  const updateCfg = useCallback((key, val) => setCfg(c => ({ ...c, [key]: val })), []);

  // ── Simulation State ───────────────────────────────────────
  const [running, setRunning] = useState(false);
  const [generation, setGeneration] = useState(0);
  const [chartData, setChartData] = useState([]);
  const [displayWorld, setDisplayWorld] = useState(null);
  const [sampleCreature, setSampleCreature] = useState(null);
  const [liveStats, setLiveStats] = useState({ survivors: 0, population: 0, diversity: 0, murders: 0, survivalPct: 0 });

  const simRef = useRef({ world: null, genomes: null, rng: null, gen: 0, running: false, cfg: null });
  const rafRef = useRef(null);

  // ── Run one generation ─────────────────────────────────────
  const runOneGen = useCallback((simState) => {
    const { world, genomes, rng, gen, cfg } = simState;

    // Build creatures
    const creatures = genomes.map(g => new Creature(0, 0, g, cfg.maxInternal));
    world.creatures = creatures;
    world.grid.fill(null);
    world.murders = 0;

    // Populate
    const positions = shuffle(Array.from({ length: world.W * world.H }, (_, i) => i), rng);
    for (let i = 0; i < Math.min(creatures.length, positions.length); i++) {
      const pos = positions[i];
      creatures[i].x = pos % world.W;
      creatures[i].y = Math.floor(pos / world.W);
      world.grid[pos] = creatures[i];
    }

    // Simulate all steps
    for (let step = 0; step < cfg.stepsPerGen; step++) {
      // Radioactivity
      if (cfg.selectionMode === "radioactive") {
        const westActive = step < cfg.stepsPerGen / 2;
        creatures.forEach(c => {
          if (!c.alive) return;
          const dist = westActive ? c.x : world.W - 1 - c.x;
          c.radiationDose += Math.exp(-0.04 * dist) * 0.012;
          if (c.radiationDose > 1.0) {
            c.alive = false;
            world.grid[c.y * world.W + c.x] = null;
          }
        });
      }
      // Shuffle and step
      for (let i = creatures.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [creatures[i], creatures[j]] = [creatures[j], creatures[i]];
      }
      creatures.forEach(c => world.stepCreature(c, step, cfg.stepsPerGen, cfg.killEnabled));
    }

    // Select
    const survivors = select(creatures, cfg.selectionMode, world.W, world.H, cfg);
    const diversity = calcDiversity(survivors);
    const pct = survivors.length / Math.max(1, creatures.length) * 100;

    // Next genomes
    simState.genomes = reproduce(survivors, cfg.population, cfg.genomeSize, cfg.mutationRate, rng);
    simState.gen = gen + 1;

    return { survivors, diversity, pct, murders: world.murders, population: creatures.length, creatures };
  }, []);

  // ── Start/stop loop ─────────────────────────────────────────
  const startSimulation = useCallback(() => {
    const rng = seededRng(Date.now());
    const world = new World(cfg.worldWidth, cfg.worldHeight);
    world.rng = rng;
    const genomes = Array.from({ length: cfg.population }, () => randomGenome(cfg.genomeSize, rng));
    simRef.current = { world, genomes, rng, gen: 0, running: true, cfg: { ...cfg } };
    setGeneration(0);
    setChartData([]);
    setRunning(true);
  }, [cfg]);

  const stopSimulation = useCallback(() => {
    simRef.current.running = false;
    setRunning(false);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
  }, []);

  const resetSimulation = useCallback(() => {
    stopSimulation();
    setGeneration(0);
    setChartData([]);
    setDisplayWorld(null);
    setSampleCreature(null);
    setLiveStats({ survivors: 0, population: 0, diversity: 0, murders: 0, survivalPct: 0 });
  }, [stopSimulation]);

  const stepOnce = useCallback(() => {
    if (!simRef.current.world) {
      const rng = seededRng(Date.now());
      const world = new World(cfg.worldWidth, cfg.worldHeight);
      world.rng = rng;
      simRef.current = { world, genomes: Array.from({ length: cfg.population }, () => randomGenome(cfg.genomeSize, rng)), rng, gen: 0, running: false, cfg: { ...cfg } };
    }
    const sim = simRef.current;
    if (sim.running) return;
    const { survivors, diversity, pct, murders, population, creatures } = runOneGen(sim);
    setGeneration(sim.gen);
    setDisplayWorld({ creatures: [...creatures], W: sim.cfg.worldWidth, H: sim.cfg.worldHeight });
    const best = survivors.length ? survivors.reduce((a, b) => a.brain.conns.length >= b.brain.conns.length ? a : b, survivors[0]) : null;
    setSampleCreature(best);
    setLiveStats({ survivors: survivors.length, population, diversity: diversity.toFixed(3), murders, survivalPct: pct.toFixed(1) });
    setChartData(prev => [...prev, { gen: sim.gen, survivors: survivors.length, diversity: diversity * 100, murders }].slice(-500));
  }, [cfg, runOneGen]);

  // ── Animation loop ─────────────────────────────────────────
  useEffect(() => {
    if (!running) return;
    const sim = simRef.current;
    if (!sim.running) return;

    const DELAYS = [300, 150, 50, 10, 0];
    const delay = DELAYS[Math.min(cfg.speed - 1, DELAYS.length - 1)];

    let timeout;
    const tick = () => {
      if (!simRef.current.running) return;
      if (simRef.current.gen >= simRef.current.cfg.maxGenerations) {
        setRunning(false);
        simRef.current.running = false;
        return;
      }
      const { survivors, diversity, pct, murders, population, creatures } = runOneGen(simRef.current);
      const gen = simRef.current.gen;
      setGeneration(gen);
      setDisplayWorld({ creatures: [...creatures], W: simRef.current.cfg.worldWidth, H: simRef.current.cfg.worldHeight });

      if (gen % 3 === 0) {
        const best = survivors.length ? survivors.reduce((a, b) => a.brain.conns.length >= b.brain.conns.length ? a : b, survivors[0]) : null;
        setSampleCreature(best);
      }
      setLiveStats({ survivors: survivors.length, population, diversity: diversity.toFixed(3), murders, survivalPct: pct.toFixed(1) });
      setChartData(prev => [...prev, { gen, survivors, diversity: +(diversity * 100).toFixed(1), murders }].slice(-500));

      if (delay === 0) rafRef.current = requestAnimationFrame(tick);
      else timeout = setTimeout(tick, delay);
    };

    if (delay === 0) rafRef.current = requestAnimationFrame(tick);
    else timeout = setTimeout(tick, delay);

    return () => { clearTimeout(timeout); if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [running, cfg.speed, runOneGen]);

  // ── Responsive canvas sizing ───────────────────────────────
  const canvasWorldRef = useRef(null);
  const canvasContainerRef = useRef(null);
  const canvasCfgRef = useRef(cfg);
  canvasCfgRef.current = cfg;

  // Keep canvas pixel dimensions in sync with its CSS container
  useEffect(() => {
    const container = canvasContainerRef.current;
    const canvas = canvasWorldRef.current;
    if (!container || !canvas) return;
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        const size = Math.min(width, height);
        canvas.width = size;
        canvas.height = size;
      }
    });
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!displayWorld) return;
    const canvas = canvasWorldRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const cw = canvas.width, ch = canvas.height;
    const W = displayWorld.W, H = displayWorld.H;
    const cellW = cw / W, cellH = ch / H;

    ctx.fillStyle = "#080c10";
    ctx.fillRect(0, 0, cw, ch);

    const mode = canvasCfgRef.current.selectionMode;
    ctx.save(); ctx.globalAlpha = 0.12; ctx.fillStyle = mode === "radioactive" ? "#ff3300" : "#00ff88";
    if (mode === "east") ctx.fillRect(cw / 2, 0, cw / 2, ch);
    else if (mode === "west") ctx.fillRect(0, 0, cw / 2, ch);
    else if (mode === "west_east") { ctx.fillRect(0, 0, cfg.stripWidth * cellW, ch); ctx.fillRect(cw - cfg.stripWidth * cellW, 0, cfg.stripWidth * cellW, ch); }
    else if (mode === "corners") { const cs = cfg.cornerSize;[[0, 0], [W - cs, 0], [0, H - cs], [W - cs, H - cs]].forEach(([rx, ry]) => ctx.fillRect(rx * cellW, ry * cellH, cs * cellW, cs * cellH)); }
    else if (mode === "center") { ctx.beginPath(); ctx.arc(cw / 2, ch / 2, cfg.centerRadius * cellW, 0, Math.PI * 2); ctx.fill(); }
    else if (mode === "radioactive") { ctx.fillRect(0, 0, W * 0.12 * cellW, ch); ctx.fillRect(cw - W * 0.12 * cellW, 0, W * 0.12 * cellW, ch); }
    ctx.restore();

    ctx.strokeStyle = "rgba(255,255,255,0.018)";
    ctx.lineWidth = 0.5;
    for (let x = 0; x <= W; x += 8) { ctx.beginPath(); ctx.moveTo(x * cellW, 0); ctx.lineTo(x * cellW, ch); ctx.stroke(); }
    for (let y = 0; y <= H; y += 8) { ctx.beginPath(); ctx.moveTo(0, y * cellH); ctx.lineTo(cw, y * cellH); ctx.stroke(); }

    displayWorld.creatures.forEach(c => {
      if (!c.alive) return;
      const px = c.x * cellW + cellW / 2, py = c.y * cellH + cellH / 2;
      const r = Math.max(1.2, cellW * 0.38);
      ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${c.color[0]},${c.color[1]},${c.color[2]})`;
      ctx.fill();
    });

    ctx.fillStyle = "rgba(0,255,136,0.45)";
    ctx.font = "bold 10px monospace";
    ctx.fillText(`GEN ${displayWorld.creatures[0]?.age !== undefined ? generation : generation}`, 8, 14);
  }, [displayWorld, generation, cfg]);

  // ── Scenario info ──────────────────────────────────────────
  const scenarioInfo = {
    east: "Survive in right half → evolves eastward migration",
    west: "Survive in left half → evolves westward migration",
    west_east: "Survive in either side strip → split instinct evolves",
    corners: "Survive in any corner → corner-seeking evolves",
    center: "Survive in centre circle → convergence evolves",
    radioactive: "Dodge alternating radioactive walls → midline crossing evolves",
    kill: "Kill neuron enabled → bi-stable violent/peaceful society",
  };

  // ── Styles ─────────────────────────────────────────────────
  const S = {
    root: {
      display: "flex", flexDirection: "column", height: "100%", background: "#080d14",
      color: "#d0e4f4", fontFamily: "'Courier New', monospace", overflow: "hidden"
    },
    header: {
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "10px 18px", borderBottom: "1px solid #1a3050", flexShrink: 0,
      background: "linear-gradient(90deg, #0a1220 0%, #0d1828 100%)"
    },
    logo: { fontSize: 18, fontWeight: "bold", color: "#00e87a", letterSpacing: "0.15em" },
    body: { display: "flex", flex: 1, overflow: "hidden", minHeight: 0 },
    sidebar: {
      width: 220, minWidth: 190, background: "#09121e", borderRight: "1px solid #1a3050",
      overflowY: "auto", padding: "12px", display: "flex", flexDirection: "column", gap: 6, flexShrink: 0
    },
    sectionTitle: {
      fontSize: 11, color: "#5a9a78", letterSpacing: "0.12em", textTransform: "uppercase",
      borderBottom: "1px solid #1a3050", paddingBottom: 5, marginBottom: 10, marginTop: 10, fontWeight: "bold"
    },
    center: { flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 0 },
    // worldBox: square canvas constrained by available height
    worldBox: {
      flex: "0 1 auto", display: "flex", justifyContent: "center", alignItems: "center",
      padding: "4px 12px", minHeight: 0
    },
    canvasContainer: {
      position: "relative", width: "100%", maxWidth: "min(100%, calc(100vh - 260px))",
      aspectRatio: "1 / 1", border: "1px solid #1a3050", borderRadius: 6,
      boxShadow: "0 0 24px rgba(0,232,122,0.06)", overflow: "hidden"
    },
    canvas: { display: "block", width: "100%", height: "100%" },
    statsRow: { display: "flex", gap: 8, padding: "0 12px 8px", flexShrink: 0 },
    statCard: {
      flex: 1, background: "#09121e", border: "1px solid #1a3050", borderRadius: 6,
      padding: "7px 10px", textAlign: "center", minWidth: 0
    },
    statVal: { fontSize: 20, color: "#00e87a", fontWeight: "bold" },
    statLbl: { fontSize: 11, color: "#6a9aaa", letterSpacing: "0.08em", textTransform: "uppercase", marginTop: 2 },
    chartBox: { flex: 1, padding: "0 12px 10px", minHeight: 0, display: "flex", flexDirection: "column" },
    right: {
      width: 290, minWidth: 230, maxWidth: 330, background: "#09121e", borderLeft: "1px solid #1a3050",
      display: "flex", flexDirection: "column", padding: 12, gap: 8, flexShrink: 0, overflowY: "auto"
    },
    btn: (col) => ({
      padding: "7px 16px", borderRadius: 4, border: `1px solid ${col}`,
      background: "transparent", color: col, cursor: "pointer", fontSize: 13,
      letterSpacing: "0.08em", fontFamily: "monospace", transition: "background 0.15s"
    }),
  };

  return (
    <div style={S.root}>
      {/* Header */}
      <div style={S.header}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <span style={S.logo}>⬡ EVOSIM</span>
          <span style={{ fontSize: 12, color: "#5a8a70", letterSpacing: "0.08em" }}>
            EVOLUTIONARY NEURAL NETWORK SIMULATOR
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <span style={{ fontSize: 13, color: "#7ab898" }}>
            {scenarioInfo[cfg.selectionMode]}
          </span>
          <span style={{ fontSize: 13, color: "#00e87a", fontWeight: "bold" }}>
            GEN {generation}/{cfg.maxGenerations}
          </span>
          <div style={{
            width: 8, height: 8, borderRadius: "50%",
            background: running ? "#00ff88" : "#334",
            boxShadow: running ? "0 0 6px #00ff88" : "none"
          }} />
        </div>
      </div>

      <div style={S.body}>
        {/* Left Sidebar – Settings */}
        <div style={S.sidebar}>
          <div style={{ ...S.sectionTitle, marginTop: 0 }}>⚙ World</div>
          <Ctrl label="Width" value={cfg.worldWidth} min={32} max={128} step={8} onChange={v => updateCfg("worldWidth", v)} />
          <Ctrl label="Height" value={cfg.worldHeight} min={32} max={128} step={8} onChange={v => updateCfg("worldHeight", v)} />
          <Ctrl label="Population" value={cfg.population} min={50} max={2000} step={50} onChange={v => updateCfg("population", v)} />
          <Ctrl label="Max Generations" value={cfg.maxGenerations} min={50} max={5000} step={50} onChange={v => updateCfg("maxGenerations", v)} />
          <Ctrl label="Steps / Generation" value={cfg.stepsPerGen} min={50} max={500} step={25} onChange={v => updateCfg("stepsPerGen", v)} />

          <div style={S.sectionTitle}>🧬 Genome & Brain</div>
          <Ctrl label="Genome Size" value={cfg.genomeSize} min={2} max={64} step={2} onChange={v => updateCfg("genomeSize", v)} />
          <Ctrl label="Internal Neurons" value={cfg.maxInternal} min={1} max={8} step={1} onChange={v => updateCfg("maxInternal", v)} />
          <Ctrl label="Mutation Rate" value={cfg.mutationRate} min={0} max={0.01} step={0.0002} onChange={v => updateCfg("mutationRate", v)} unit="" />

          <div style={S.sectionTitle}>🎯 Selection</div>
          <Select label="Scenario" value={cfg.selectionMode}
            options={[
              { value: "east", label: "East Half" },
              { value: "west", label: "West Half" },
              { value: "west_east", label: "Both Sides" },
              { value: "corners", label: "Corners" },
              { value: "center", label: "Centre Circle" },
              { value: "radioactive", label: "Radioactive Walls" },
              { value: "kill", label: "Kill Enabled" },
            ]}
            onChange={v => updateCfg("selectionMode", v)} />

          {cfg.selectionMode === "center" &&
            <Ctrl label="Centre Radius" value={cfg.centerRadius} min={5} max={40} step={1} onChange={v => updateCfg("centerRadius", v)} />}
          {cfg.selectionMode === "west_east" &&
            <Ctrl label="Strip Width" value={cfg.stripWidth} min={8} max={48} step={4} onChange={v => updateCfg("stripWidth", v)} />}
          {cfg.selectionMode === "corners" &&
            <Ctrl label="Corner Size" value={cfg.cornerSize} min={8} max={48} step={4} onChange={v => updateCfg("cornerSize", v)} />}

          <Toggle label="Kill Neuron" value={cfg.killEnabled} onChange={v => updateCfg("killEnabled", v)} />

          <div style={S.sectionTitle}>▶ Playback</div>
          <Ctrl label="Speed" value={cfg.speed} min={1} max={5} step={1} onChange={v => updateCfg("speed", v)} />
          <div style={{ fontSize: 11, color: "#6a9a80", marginTop: -4 }}>
            {["Slow", "Medium", "Fast", "Turbo", "Max"][cfg.speed - 1]}
          </div>
        </div>

        {/* Centre – World + Stats + Chart */}
        <div style={S.center}>
          {/* Control buttons */}
          <div style={{ display: "flex", gap: 8, padding: "10px 12px 4px" }}>
            {!running
              ? <button style={S.btn("#00ff88")} onClick={startSimulation}>▶ START</button>
              : <button style={S.btn("#ff4455")} onClick={stopSimulation}>⏸ PAUSE</button>}
            <button style={S.btn("#4499ff")} onClick={stepOnce} disabled={running}>⏭ STEP</button>
            <button style={S.btn("#888")} onClick={resetSimulation}>↺ RESET</button>
          </div>

          {/* World canvas – fills a square container sized by CSS aspect-ratio */}
          <div style={S.worldBox}>
            <div ref={canvasContainerRef} style={S.canvasContainer}>
              <canvas ref={canvasWorldRef} style={S.canvas} />
            </div>
          </div>

          {/* Live stats */}
          <div style={S.statsRow}>
            {[
              { val: liveStats.survivalPct + "%", lbl: "Survival" },
              { val: liveStats.survivors, lbl: "Survivors" },
              { val: liveStats.diversity, lbl: "Diversity" },
              { val: liveStats.murders, lbl: "Murders" },
            ].map(({ val, lbl }) => (
              <div key={lbl} style={S.statCard}>
                <div style={S.statVal}>{val}</div>
                <div style={S.statLbl}>{lbl}</div>
              </div>
            ))}
          </div>

          {/* Evolution chart */}
          <div style={S.chartBox}>
            <div style={{ fontSize: 12, color: "#5a9a78", letterSpacing: "0.1em", marginBottom: 6, fontWeight: "bold" }}>
              EVOLUTION CHART
            </div>
            {chartData.length > 1 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 4, right: 10, bottom: 4, left: 0 }}>
                  <XAxis dataKey="gen" tick={{ fill: "#7a9ab8", fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis yAxisId="left" tick={{ fill: "#7a9ab8", fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fill: "#7a9ab8", fontSize: 11 }} tickLine={false} axisLine={false} domain={[0, 100]} />
                  <Tooltip contentStyle={{ background: "#0d1828", border: "1px solid #1a3050", fontSize: 12, color: "#d0e4f4" }} />
                  <Line yAxisId="left" type="monotone" dataKey="survivors" stroke="#00e87a" dot={false} strokeWidth={2} name="Survivors" />
                  <Line yAxisId="right" type="monotone" dataKey="diversity" stroke="#bb66ff" dot={false} strokeWidth={1.5} strokeDasharray="4 3" name="Diversity %" />
                  {chartData.some(d => d.murders > 0) &&
                    <Line yAxisId="left" type="monotone" dataKey="murders" stroke="#ff7744" dot={false} strokeWidth={1.5} name="Murders" />}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div style={{
                flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
                color: "#5a7a90", fontSize: 13, border: "1px dashed #1a3050", borderRadius: 4
              }}>
                Press START to begin simulation
              </div>
            )}
          </div>
        </div>

        {/* Right panel – Neural Network */}
        <div style={S.right}>
          <div style={{ fontSize: 12, color: "#5a9a78", letterSpacing: "0.1em", marginBottom: 6, fontWeight: "bold" }}>
            NEURAL NETWORK — BEST SURVIVOR
          </div>
          <div style={{
            flex: 1, border: "1px solid #1a3050", borderRadius: 6, background: "#060d16",
            overflow: "hidden", minHeight: 200
          }}>
            <NeuralDiagram creature={sampleCreature} />
          </div>

          {sampleCreature && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 12, color: "#5a9a78", letterSpacing: "0.1em", marginBottom: 6, fontWeight: "bold" }}>
                GENOME SAMPLE
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
                {sampleCreature.genome.slice(0, 16).map((g, i) => (
                  <span key={i} style={{
                    fontSize: 10, fontFamily: "monospace",
                    color: "#6ab898", background: "#0d1a28", padding: "2px 4px", borderRadius: 3
                  }}>
                    {(g >>> 0).toString(16).padStart(8, "0")}
                  </span>
                ))}
              </div>
            </div>
          )}

          {sampleCreature && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 12, color: "#5a9a78", letterSpacing: "0.1em", marginBottom: 6, fontWeight: "bold" }}>
                ACTIVE CONNECTIONS
              </div>
              <div style={{ maxHeight: 180, overflowY: "auto" }}>
                {sampleCreature.brain.conns.slice(0, 20).map((c, i) => {
                  const src = c.sourceType === 0 ? SENSOR_LABELS[c.sourceId] : `internal_${c.sourceId}`;
                  const snk = c.sinkType === 1 ? ACTION_LABELS[c.sinkId] : `internal_${c.sinkId}`;
                  return (
                    <div key={i} style={{
                      fontSize: 11, display: "flex", justifyContent: "space-between",
                      padding: "3px 0", borderBottom: "1px solid #0d1a28", gap: 6
                    }}>
                      <span style={{ color: "#6699dd", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{src}</span>
                      <span style={{ color: c.weight >= 0 ? "#33cc77" : "#ee4455", minWidth: 44, textAlign: "right", fontWeight: "bold" }}>
                        {c.weight >= 0 ? "+" : ""}{c.weight.toFixed(2)}
                      </span>
                      <span style={{ color: "#dd7799", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{snk}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <div style={{ marginTop: "auto", padding: "10px 0 0", borderTop: "1px solid #1a3050" }}>
            <div style={{ fontSize: 11, color: "#8ab0c0", lineHeight: 1.8 }}>
              <div style={{ color: "#00e87a", display: "inline" }}>■ </div>Survivors (green zone)<br />
              <div style={{ color: "#bb66ff", display: "inline" }}>■ </div>Genetic diversity<br />
              <div style={{ color: "#5599ee", display: "inline" }}>● </div>Sensor neurons<br />
              <div style={{ color: "#ff88aa", display: "inline" }}>● </div>Action neurons
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
