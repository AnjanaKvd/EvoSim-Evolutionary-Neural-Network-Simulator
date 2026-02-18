"""
Fast verification demo – 100 generations, 200 pop
Runs in ~60 seconds and saves all outputs.
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
    if gen_idx % 20 == 0:
        print(f"  Saving snapshot gen {gen_idx}...")
        save_world_snapshot(world, gen_idx, survivors, "east", OUT)
        if survivors:
            best = max(survivors, key=lambda c: len(c.brain.get_active_connections()))
            save_neural_diagram(best, gen_idx, "best", OUT)

sim = Simulation(
    population      = 200,
    max_generations = 100,
    steps_per_gen   = 150,
    genome_size     = 8,
    mutation_rate   = 0.001,
    selection_mode  = "east",
    seed            = 42,
    on_gen_callback = on_gen,
)
sim.run()
save_evolution_chart(all_stats, OUT, "demo_chart.png")

# Save final snapshot
survivors_final = [c for c in sim.current_gen_creatures if c.alive]
save_world_snapshot(sim.world, sim.generation, survivors_final, "east", OUT)
print("\nAll outputs in:", OUT)
print("Files:")
for root, dirs, files in os.walk(OUT):
    for f in files:
        path = os.path.join(root, f)
        print(f"  {path}")
