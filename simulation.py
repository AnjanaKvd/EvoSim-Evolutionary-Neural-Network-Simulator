"""
Simulation Engine for EvoSim.

Orchestrates the full evolutionary loop:
  for each generation:
    1. (Re)populate world from current population
    2. Run STEPS_PER_GEN simulator steps
    3. Apply selection criterion → survivors
    4. Reproduce survivors → next generation
    5. Log stats
"""

import numpy as np
import time
from world import World
from creature import Creature
from genome import (random_genome, crossover, mutate_genome,
                    genome_similarity)
from config import (
    WORLD_WIDTH, WORLD_HEIGHT, POPULATION,
    MAX_GENERATIONS, STEPS_PER_GEN, GENOME_SIZE,
    MUTATION_RATE, SELECTION_MODE,
    CENTER_RADIUS, STRIP_WIDTH, CORNER_SIZE,
    RADIO_MAX_DOSE, RADIO_FALLOFF, KILL_ENABLED,
)


class Simulation:
    """
    Main simulation controller.
    """

    def __init__(
        self,
        population:       int  = POPULATION,
        max_generations:  int  = MAX_GENERATIONS,
        steps_per_gen:    int  = STEPS_PER_GEN,
        genome_size:      int  = GENOME_SIZE,
        mutation_rate:    float= MUTATION_RATE,
        selection_mode:   str  = SELECTION_MODE,
        world_width:      int  = WORLD_WIDTH,
        world_height:     int  = WORLD_HEIGHT,
        seed:             int  = None,
        on_step_callback  = None,    # called every sim step (for live viz)
        on_gen_callback   = None,    # called at end of each generation
    ):
        self.population      = population
        self.max_generations = max_generations
        self.steps_per_gen   = steps_per_gen
        self.genome_size     = genome_size
        self.mutation_rate   = mutation_rate
        self.selection_mode  = selection_mode
        self.world           = World(world_width, world_height, seed)
        self.rng             = self.world.rng
        self.on_step_callback = on_step_callback
        self.on_gen_callback  = on_gen_callback

        # History
        self.generation  = 0
        self.stats       = []          # list of dicts, one per generation
        self.current_gen_creatures = []

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(self):
        """Run the full simulation for max_generations generations."""
        # Seed generation 0 with random creatures
        genomes = [random_genome(self.genome_size, self.rng)
                   for _ in range(self.population)]
        self._run_all_generations(genomes)

    def _run_all_generations(self, initial_genomes):
        genomes = initial_genomes
        for gen_idx in range(self.max_generations):
            self.generation = gen_idx
            t0 = time.time()

            survivors, stats = self._run_one_generation(genomes)

            elapsed = time.time() - t0
            stats["elapsed_s"] = round(elapsed, 3)
            self.stats.append(stats)

            self._print_stats(gen_idx, stats)

            if self.on_gen_callback:
                self.on_gen_callback(gen_idx, stats, self.world,
                                     self.current_gen_creatures, survivors)

            if not survivors:
                print("  !! Extinction event – all creatures died. Stopping.")
                break

            # Breed next generation
            genomes = self._reproduce(survivors)

        print("\n=== Simulation complete ===")

    # ──────────────────────────────────────────────────────────────────────────
    # One generation
    # ──────────────────────────────────────────────────────────────────────────

    def _run_one_generation(self, genomes: list):
        """
        Create creatures from genomes, run them for steps_per_gen steps,
        apply selection, and return (survivors, stats_dict).
        """
        # Build creatures
        creatures = [
            Creature(0, 0, genome, self.rng) for genome in genomes
        ]
        self.current_gen_creatures = creatures

        # Place on world
        self.world.populate(creatures)
        self.world.murder_count = 0

        # Simulation loop
        for step in range(self.steps_per_gen):
            # Radioactivity damage (if applicable)
            if self.selection_mode == "radioactive":
                self._apply_radiation(creatures, step)

            # Each creature takes one step
            order = self.rng.permutation(len(creatures))
            for idx in order:
                c = creatures[idx]
                if c.alive:
                    c.step(self.world, step)

            if self.on_step_callback:
                self.on_step_callback(step, self.world, creatures)

        # Apply selection
        survivors = self._select(creatures)

        # Compute stats
        stats = self._compute_stats(creatures, survivors)
        return survivors, stats

    # ──────────────────────────────────────────────────────────────────────────
    # Selection criteria
    # ──────────────────────────────────────────────────────────────────────────

    def _select(self, creatures: list) -> list:
        mode = self.selection_mode
        W, H = self.world.width, self.world.height
        survivors = []

        for c in creatures:
            if not c.alive:
                continue
            if mode == "east":
                if c.x >= W // 2:
                    survivors.append(c)

            elif mode == "west":
                if c.x < W // 2:
                    survivors.append(c)

            elif mode == "west_east":
                if c.x < STRIP_WIDTH or c.x >= W - STRIP_WIDTH:
                    survivors.append(c)

            elif mode == "corners":
                if (c.x < CORNER_SIZE or c.x >= W - CORNER_SIZE) and \
                   (c.y < CORNER_SIZE or c.y >= H - CORNER_SIZE):
                    survivors.append(c)

            elif mode == "center":
                cx, cy = W // 2, H // 2
                if (c.x - cx)**2 + (c.y - cy)**2 <= CENTER_RADIUS**2:
                    survivors.append(c)

            elif mode == "radioactive":
                # Survive if still alive (radiation has already killed some)
                survivors.append(c)

            else:
                # Default: everyone survives (no selection pressure)
                survivors.append(c)

        return survivors

    # ──────────────────────────────────────────────────────────────────────────
    # Radiation
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_radiation(self, creatures: list, step: int):
        """
        First half: west wall radiates.  Second half: east wall radiates.
        Dose accumulates; creatures die if dose > threshold.
        """
        W = self.world.width
        west_active = step < self.steps_per_gen // 2

        for c in creatures:
            if not c.alive:
                continue
            if west_active:
                dist = c.x
            else:
                dist = W - 1 - c.x
            # Exponential falloff from wall
            dose_per_step = np.exp(-RADIO_FALLOFF * dist) * 0.01
            c.radiation_dose += dose_per_step
            if c.radiation_dose > RADIO_MAX_DOSE:
                c.alive = False
                self.world.remove_creature(c)

    # ──────────────────────────────────────────────────────────────────────────
    # Reproduction
    # ──────────────────────────────────────────────────────────────────────────

    def _reproduce(self, survivors: list) -> list:
        """
        Produce exactly self.population new genomes from survivors.
        Uses random pairing + crossover + mutation.
        """
        if not survivors:
            # Full random restart if extinction
            return [random_genome(self.genome_size, self.rng)
                    for _ in range(self.population)]

        new_genomes = []
        n = len(survivors)
        while len(new_genomes) < self.population:
            # Pick two parents randomly (with replacement)
            pa = survivors[int(self.rng.integers(0, n))]
            pb = survivors[int(self.rng.integers(0, n))]
            child_genome = crossover(pa.genome, pb.genome, self.rng)
            child_genome = mutate_genome(child_genome, self.mutation_rate, self.rng)
            new_genomes.append(child_genome)

        return new_genomes[:self.population]

    # ──────────────────────────────────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_stats(self, creatures: list, survivors: list) -> dict:
        n_pop       = len(creatures)
        n_survivors = len(survivors)
        n_alive     = sum(1 for c in creatures if c.alive)
        n_murdered  = self.world.murder_count

        # Genetic diversity: average pairwise Hamming distance (sampled)
        diversity = self._genetic_diversity(survivors)

        return {
            "generation":   self.generation,
            "population":   n_pop,
            "survivors":    n_survivors,
            "alive":        n_alive,
            "survival_pct": 100.0 * n_survivors / max(1, n_pop),
            "murdered":     n_murdered,
            "diversity":    diversity,
        }

    def _genetic_diversity(self, creatures: list, sample: int = 50) -> float:
        """
        Estimate genetic diversity as average pairwise dissimilarity.
        Returns value 0 (identical) → 1 (maximally diverse).
        """
        if len(creatures) < 2:
            return 0.0
        sample_size = min(sample, len(creatures))
        idx = self.rng.choice(len(creatures), sample_size, replace=False)
        sampled = [creatures[i] for i in idx]
        total, count = 0.0, 0
        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                sim = genome_similarity(sampled[i].genome, sampled[j].genome)
                total += (1.0 - sim)
                count += 1
        return total / count if count else 0.0

    def _print_stats(self, gen_idx: int, stats: dict):
        if gen_idx % 10 == 0 or gen_idx < 5:
            print(
                f"Gen {gen_idx:>5}  |  "
                f"survivors {stats['survivors']:>5}/{stats['population']:<5}"
                f"({stats['survival_pct']:>5.1f}%)  |  "
                f"diversity {stats['diversity']:.3f}  |  "
                f"murders {stats['murdered']:>4}  |  "
                f"{stats['elapsed_s']:.2f}s"
            )
