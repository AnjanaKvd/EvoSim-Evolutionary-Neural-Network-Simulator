"""
Quick demo – runs a 200-generation east-half simulation
and saves snapshots + charts without needing a display.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from simulation import Simulation
from visualizer import (ensure_dirs, save_world_snapshot,
                         save_evolution_chart, save_neural_diagram, append_csv)

OUT = "output/demo"
ensure_dirs(OUT)

all_stats = []

def on_gen(gen_idx, stats, world, creatures, survivors):
    all_stats.append(stats)
    append_csv(stats, OUT)
    if gen_idx % 25 == 0:
        save_world_snapshot(world, gen_idx, survivors, "east", OUT)
        if survivors:
            best = max(survivors, key=lambda c: len(c.brain.get_active_connections()))
            save_neural_diagram(best, gen_idx, "best", OUT)

sim = Simulation(
    population      = 500,
    max_generations = 200,
    steps_per_gen   = 200,
    genome_size     = 8,
    mutation_rate   = 0.001,
    selection_mode  = "east",
    seed            = 42,
    on_gen_callback = on_gen,
)
sim.run()

save_evolution_chart(all_stats, OUT, "demo_chart.png")
print("\nAll outputs in:", OUT)
