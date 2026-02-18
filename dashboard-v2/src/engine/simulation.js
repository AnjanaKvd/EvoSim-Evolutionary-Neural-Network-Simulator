/* ============================================================
   SIMULATION MODULE
   ============================================================ */
import { Creature } from './creature';
import { World } from './world';
import { randomGenome, reproduce, calcDiversity } from './genome';
import { shuffle, seededRng } from './utils';

export function select(creatures, mode, W, H, cfg) {
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

// runOneGen: Pure function to run one generation
// Returns stats and new state
export function runOneGen(simState) {
    const { world, genomes, rng, gen, cfg } = simState;

    // Build creatures
    // Pass rng so lastDir is seeded (not Math.random)
    const creatures = genomes.map(g => new Creature(0, 0, g, cfg.maxInternal, rng));
    world.populate(creatures);

    // Simulate all steps
    for (let step = 0; step < cfg.stepsPerGen; step++) {
        // Radioactivity — applied before movement (same as Python)
        if (cfg.selectionMode === "radioactive") {
            const westActive = step < cfg.stepsPerGen / 2;
            creatures.forEach(c => {
                if (!c.alive) return;
                const dist = westActive ? c.x : world.W - 1 - c.x;
                // Match Python: exp(-0.04 * dist) * 0.03 (moderate dose)
                c.radiationDose += Math.exp(-0.04 * dist) * 0.03;
                if (c.radiationDose > 1.0) {
                    c.alive = false;
                    world.grid[c.y * world.W + c.x] = null;
                }
            });
        }

        // Shuffle order of updates
        shuffle(creatures, rng);

        // Step each creature
        creatures.forEach(c => world.stepCreature(c, step, cfg.stepsPerGen, cfg.killEnabled));
    }

    // Select
    const survivors = select(creatures, cfg.selectionMode, world.W, world.H, cfg);
    const diversity = calcDiversity(survivors);
    const pct = survivors.length / Math.max(1, creatures.length) * 100;

    // Next genomes
    const nextGenomes = reproduce(survivors, cfg.population, cfg.genomeSize, cfg.mutationRate, rng);

    // Update state
    simState.genomes = nextGenomes;
    simState.gen = gen + 1;

    return {
        survivors,
        diversity,
        pct,
        murders: world.murders,
        population: creatures.length,
        creatures
    };
}

export function createSimulation(cfg) {
    const rng = seededRng(Date.now());
    const world = new World(cfg.worldWidth, cfg.worldHeight);
    world.rng = rng;
    const genomes = Array.from({ length: cfg.population }, () => randomGenome(cfg.genomeSize, rng));

    return {
        world,
        genomes,
        rng,
        gen: 0,
        running: false,
        cfg: { ...cfg }
    };
}
