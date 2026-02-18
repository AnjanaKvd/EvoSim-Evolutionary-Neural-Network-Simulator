import { useState, useRef, useEffect, useCallback } from 'react';
import { Layout, Button } from './components/Layout';
import { SidebarControls } from './components/SidebarControls';
import { StatsPanel } from './components/StatsPanel';
import { WorldRenderer } from './components/WorldRenderer';
import { BrainView } from './components/BrainView';
import { createSimulation, runOneGen } from './engine/simulation';
import { Play, Pause, RotateCcw, FastForward } from 'lucide-react';

function App() {
  // Config State
  const [config, setConfig] = useState({
    worldWidth: 96, worldHeight: 96,
    population: 400, maxGenerations: 500,
    stepsPerGen: 200, genomeSize: 12,
    maxInternal: 4, mutationRate: 0.001,
    selectionMode: "east",
    stripWidth: 24, cornerSize: 24,
    centerRadius: 28,
    killEnabled: false,
    speed: 3,
  });

  // Simulation State
  const simRef = useRef(createSimulation(config));
  const [generation, setGeneration] = useState(0);
  const [running, setRunning] = useState(false);
  const [stats, setStats] = useState({ survivors: 0, population: 0, diversity: 0, murders: 0, survivalPct: 0 });
  const [history, setHistory] = useState([]);
  const [displayWorld, setDisplayWorld] = useState(null); // Snapshot for rendering
  const [selectedCreature, setSelectedCreature] = useState(null);

  const rafRef = useRef(null);

  // Initialize
  useEffect(() => {
    // Initial render setup
    const sim = simRef.current;
    setDisplayWorld({
      creatures: [...sim.world.creatures],
      W: sim.world.W,
      H: sim.world.H
    });
  }, []);

  // Sync config changes to running simulation
  useEffect(() => {
    if (simRef.current) {
      simRef.current.cfg = { ...config };
    }
  }, [config]);

  // Control Handlers
  const handleStart = () => {
    if (!simRef.current.running) {
      simRef.current.running = true;
      setRunning(true);
    }
  };

  const handleStop = () => {
    simRef.current.running = false;
    setRunning(false);
  };

  const handleReset = () => {
    handleStop();
    const newSim = createSimulation(config);
    simRef.current = newSim;
    setGeneration(0);
    setHistory([]);
    setStats({ survivors: 0, population: 0, diversity: 0, murders: 0, survivalPct: 0 });
    setSelectedCreature(null);
    setDisplayWorld({ creatures: [...newSim.world.creatures], W: newSim.world.W, H: newSim.world.H });
  };

  // Loop
  const tick = useCallback(() => {
    const sim = simRef.current;
    if (!sim.running) return;

    if (sim.gen >= sim.cfg.maxGenerations) {
      handleStop();
      return;
    }

    const res = runOneGen(sim);

    // Update State
    setGeneration(sim.gen);
    setStats({
      survivors: res.survivors.length,
      population: res.population,
      diversity: res.diversity.toFixed(3),
      murders: res.murders,
      survivalPct: res.pct.toFixed(1)
    });
    setHistory(prev => [...prev, { gen: sim.gen, survivors: res.survivors.length, diversity: (res.diversity * 100).toFixed(1) }].slice(-100)); // Keep last 100 for smoother chart
    setDisplayWorld({ creatures: res.creatures, W: sim.world.W, H: sim.world.H });

    // Auto-select best creature occasionally
    if (sim.gen % 5 === 0 && res.survivors.length > 0) {
      // Simple heuristic: most complex brain
      const best = res.survivors.reduce((a, b) => a.brain.conns.length >= b.brain.conns.length ? a : b, res.survivors[0]);
      setSelectedCreature(best);
    }

    // Speed Control
    const delay = [500, 200, 50, 0][config.speed] ?? 0; // 0, 1, 2, 3 maps to Slow, Med, Fast, Max

    if (delay === 0) {
      rafRef.current = requestAnimationFrame(tick);
    } else {
      setTimeout(() => {
        // Re-check running state in timeout
        if (simRef.current.running) tick();
      }, delay);
    }

  }, [config.speed]); // depends on speed config

  useEffect(() => {
    if (running) {
      tick();
    }
    return () => cancelAnimationFrame(rafRef.current);
  }, [running, tick]);


  // Prepare Header Controls
  const headerControls = (
    <>
      <div className="flex bg-[var(--bg-dark)] rounded border border-[var(--border)] p-1 gap-1">
        <Button icon={Play} onClick={handleStart} active={running} label="RUN" />
        <Button icon={Pause} onClick={handleStop} active={!running && generation > 0} label="PAUSE" />
        <Button icon={RotateCcw} onClick={handleReset} label="RESET" danger />
      </div>
      <div className="flex items-center gap-2 ml-4">
        <span className="text-[10px] uppercase font-bold text-[var(--text-muted)]">SPEED</span>
        <div className="flex gap-1">
          {[0, 1, 2, 3].map(s => (
            <div
              key={s}
              onClick={() => setConfig(c => ({ ...c, speed: s }))}
              className={`w-8 h-4 rounded cursor-pointer ${config.speed >= s ? 'bg-[var(--primary)] box-shadow-[0_0_5px_var(--primary)]' : 'bg-[var(--border)]'}`}
            />
          ))}
        </div>
      </div>
    </>
  );

  return (
    <Layout
      header={headerControls}
      sidebar={<SidebarControls config={config} setConfig={setConfig} />}
      stats={
        <div className="flex flex-col gap-4 h-full">
          <StatsPanel stats={stats} history={history} />
          <div className="flex-1 min-h-0 flex flex-col">
            <div className="p-3 border-b border-[var(--border)] uppercase text-xs font-bold text-[var(--text-muted)] tracking-wider">
              Specimen Analysis
            </div>
            <div className="p-4 flex-1">
              <BrainView creature={selectedCreature} />
              <div className="mt-4 text-xs font-mono text-[var(--text-muted)]">
                {selectedCreature ? (
                  <>
                    <div className="flex justify-between"><span>GENOME ID:</span> <span className="text-white">{selectedCreature.genome.slice(0, 4).join('')}...</span></div>
                    <div className="flex justify-between"><span>AGE:</span> <span className="text-white">{selectedCreature.age}</span></div>
                    <div className="flex justify-between"><span>SYNAPSES:</span> <span className="text-white">{selectedCreature.brain.conns.length}</span></div>
                  </>
                ) : "Select a creature (Auto-selection active)"}
              </div>
            </div>
          </div>
        </div>
      }
    >
      <div className="w-full h-full p-4 flex items-center justify-center">
        {/* Canvas Container */}
        <div className="w-full h-full max-w-[80vh] max-h-[80vh] aspect-square relative">
          {displayWorld && <WorldRenderer world={displayWorld} generation={generation} config={config} />}
        </div>
      </div>
    </Layout>
  );
}

export default App;
