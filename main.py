"""
EvoSim – Main Entry Point
=========================

Usage examples:
  python main.py                          # default: east-half selection
  python main.py --scenario west_east     # survive in left or right strip
  python main.py --scenario corners       # survive in any corner
  python main.py --scenario center        # survive in center circle
  python main.py --scenario radioactive   # avoid radioactive walls
  python main.py --scenario kill          # enable kill neuron experiment
  python main.py --gens 500 --pop 500     # custom parameters
  python main.py --brain_size 24          # larger brains (24 genes)
  python main.py --no_mutation            # turn off mutations (demonstration)
"""

import argparse
import os
import sys

from simulation  import Simulation
from visualizer  import (ensure_dirs, save_world_snapshot,
                          save_evolution_chart, save_neural_diagram,
                          append_csv)
from config import (SAVE_DIR, SNAPSHOT_INTERVAL, SAVE_NEURAL_SAMPLE,
                    POPULATION, MAX_GENERATIONS, STEPS_PER_GEN,
                    GENOME_SIZE, MUTATION_RATE)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="EvoSim – Evolutionary Neural Network Simulator")
    p.add_argument("--scenario",    default="east",
                   choices=["east","west","west_east","corners",
                             "center","radioactive","kill"],
                   help="Selection / survival scenario")
    p.add_argument("--gens",       type=int,   default=MAX_GENERATIONS,
                   help="Number of generations to run")
    p.add_argument("--pop",        type=int,   default=POPULATION,
                   help="Population size")
    p.add_argument("--steps",      type=int,   default=STEPS_PER_GEN,
                   help="Simulator steps per generation")
    p.add_argument("--brain_size", type=int,   default=GENOME_SIZE,
                   help="Genome size = max neural connections")
    p.add_argument("--mutation",   type=float, default=MUTATION_RATE,
                   help="Mutation rate per bit")
    p.add_argument("--no_mutation",action="store_true",
                   help="Set mutation rate to 0 (demonstration)")
    p.add_argument("--seed",       type=int,   default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--outdir",     default=SAVE_DIR,
                   help="Output directory")
    p.add_argument("--snapshot_interval", type=int, default=SNAPSHOT_INTERVAL,
                   help="Save world snapshot every N generations")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────

class SimCallbacks:
    """Bundles the per-generation callbacks used by the simulation."""

    def __init__(self, outdir: str, snapshot_interval: int,
                 all_stats: list, selection_mode: str):
        self.outdir           = outdir
        self.snapshot_interval = snapshot_interval
        self.all_stats        = all_stats
        self.selection_mode   = selection_mode

    def on_generation(self, gen_idx, stats, world, creatures, survivors):
        # Append to stats list
        self.all_stats.append(stats)

        # CSV log
        append_csv(stats, self.outdir)

        # Save snapshot
        if gen_idx % self.snapshot_interval == 0:
            path = save_world_snapshot(
                world, gen_idx, survivors,
                self.selection_mode, self.outdir)
            print(f"  → Snapshot: {path}")

            # Neural network diagram of one surviving creature
            if SAVE_NEURAL_SAMPLE and survivors:
                # Pick the creature with the most active connections
                best = max(survivors,
                           key=lambda c: len(c.brain.get_active_connections()))
                npath = save_neural_diagram(
                    best, gen_idx, "best_survivor", self.outdir)
                if npath:
                    print(f"  → Neural diagram: {npath}")

        # Final chart update every 100 gens
        if gen_idx % 100 == 0 and gen_idx > 0:
            save_evolution_chart(self.all_stats, self.outdir)


# ──────────────────────────────────────────────────────────────────────────────
# Scenario descriptions
# ──────────────────────────────────────────────────────────────────────────────

SCENARIO_DESCRIPTIONS = {
    "east": (
        "Eastern half survival.\n"
        "Creatures that end up in the right half of the world survive.\n"
        "Expected: creatures evolve to migrate east."
    ),
    "west": (
        "Western half survival.\n"
        "Creatures that end up in the left half survive.\n"
        "Expected: creatures evolve to migrate west."
    ),
    "west_east": (
        "Side strips survival.\n"
        "Creatures must reach either the left or right edge strip.\n"
        "Expected: split population – some go east, some west."
    ),
    "corners": (
        "Corner survival.\n"
        "Creatures must reach any of the four corner regions.\n"
        "Expected: creatures evolve to find a corner and stay there."
    ),
    "center": (
        "Central sanctuary.\n"
        "Only creatures in the centre circle survive.\n"
        "Expected: creatures evolve to converge on the centre."
    ),
    "radioactive": (
        "Radioactive walls challenge.\n"
        "First half of lifetime: west wall radiates.\n"
        "Second half: east wall radiates.\n"
        "Expected: creatures learn to dodge the hot wall."
    ),
    "kill": (
        "Kill neuron enabled.\n"
        "Creatures can kill adjacent neighbours.\n"
        "Expected: bi-stable peaceful/violent society."
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    outdir = os.path.join(args.outdir, args.scenario)
    ensure_dirs(outdir)

    mutation_rate = 0.0 if args.no_mutation else args.mutation

    print("=" * 60)
    print("  EvoSim – Evolutionary Neural Network Simulator")
    print("=" * 60)
    print(f"  Scenario   : {args.scenario}")
    print(f"  Description: {SCENARIO_DESCRIPTIONS.get(args.scenario, '')}")
    print(f"  Population : {args.pop}")
    print(f"  Generations: {args.gens}")
    print(f"  Steps/gen  : {args.steps}")
    print(f"  Genome size: {args.brain_size} genes")
    print(f"  Mutation   : {mutation_rate}")
    print(f"  Output dir : {outdir}")
    print("=" * 60)

    # Enable kill neuron if requested
    if args.scenario == "kill":
        import config
        config.KILL_ENABLED = True

    all_stats = []

    cb = SimCallbacks(
        outdir           = outdir,
        snapshot_interval = args.snapshot_interval,
        all_stats        = all_stats,
        selection_mode   = args.scenario if args.scenario != "kill" else "center",
    )

    sim = Simulation(
        population      = args.pop,
        max_generations = args.gens,
        steps_per_gen   = args.steps,
        genome_size     = args.brain_size,
        mutation_rate   = mutation_rate,
        selection_mode  = args.scenario if args.scenario != "kill" else "center",
        seed            = args.seed,
        on_gen_callback = cb.on_generation,
    )

    sim.run()

    # Final chart
    print("\nSaving final evolution chart …")
    chart_path = save_evolution_chart(all_stats, outdir, "evolution_final.png")
    print(f"  → {chart_path}")

    # Final world snapshot
    if sim.current_gen_creatures:
        survivors = [c for c in sim.current_gen_creatures if c.alive]
        snap = save_world_snapshot(
            sim.world, sim.generation, survivors,
            sim.selection_mode, outdir)
        print(f"  → Final snapshot: {snap}")

    print("\nDone! All outputs saved to:", outdir)


if __name__ == "__main__":
    main()
