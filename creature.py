"""
Creature class for EvoSim.

Each creature has:
  - (x, y) position on the grid
  - A genome (list of int32 genes)
  - A NeuralNetwork brain built from the genome
  - State: age, last_dir, alive, radiation_dose

Every simulation step the creature:
  1. Gathers sensor readings from the world
  2. Runs its neural network
  3. Executes the strongest action output
"""

import numpy as np
from genome import random_genome, genome_to_color
from neural_network import NeuralNetwork
from config import (
    NUM_SENSORS, NUM_ACTIONS, GENOME_SIZE,
    STEPS_PER_GEN, KILL_ENABLED,
    WORLD_WIDTH, WORLD_HEIGHT,
)

# 8 compass directions (dx, dy)  – index 0-7
DIRS = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]


class Creature:
    """
    A single agent in the evolutionary simulation.
    """
    __slots__ = (
        "x", "y", "genome", "brain", "alive",
        "age", "last_dir", "color",
        "radiation_dose", "_osc_phase"
    )

    def __init__(self, x: int, y: int, genome: list = None, rng=None):
        self.x = x
        self.y = y
        self.genome    = genome if genome is not None else random_genome(GENOME_SIZE, rng)
        self.brain     = NeuralNetwork(self.genome)
        self.alive     = True
        self.age       = 0                  # steps lived this generation
        self.last_dir  = 0                  # index into DIRS
        self.color     = genome_to_color(self.genome)
        self.radiation_dose = 0.0
        self._osc_phase    = 0.0            # oscillator phase (radians)

    # ──────────────────────────────────────────────────────────────────────────

    def _sense(self, world, sim_step: int) -> np.ndarray:
        """Compute all sensory input values (0..1 range)."""
        W, H = world.width, world.height
        inputs = np.zeros(NUM_SENSORS, dtype=np.float32)

        # 0: loc_x
        inputs[0] = self.x / (W - 1)
        # 1: loc_y
        inputs[1] = self.y / (H - 1)
        # 2: age
        inputs[2] = self.age / max(1, STEPS_PER_GEN - 1)
        # 3: random noise
        inputs[3] = world.rng.random()
        # 4: oscillator (sin, period ~30 steps)
        self._osc_phase += 2 * np.pi / 30.0
        inputs[4] = (np.sin(self._osc_phase) + 1.0) * 0.5
        # 5: distance to nearest EW wall  (1 = at wall, 0 = center)
        inputs[5] = 1.0 - (min(self.x, W - 1 - self.x) / (W / 2))
        # 6: distance to nearest NS wall
        inputs[6] = 1.0 - (min(self.y, H - 1 - self.y) / (H / 2))
        # 7: local population density (3x3 neighbourhood, excluding self)
        neighbours = world.local_density(self.x, self.y, radius=2)
        inputs[7] = min(1.0, neighbours / 8.0)
        # 8: population gradient in forward direction
        inputs[8] = world.forward_density_gradient(self.x, self.y, self.last_dir)
        # 9: genetic similarity to creature directly ahead
        dx, dy = DIRS[self.last_dir]
        nx, ny = self.x + dx, self.y + dy
        ahead  = world.get_creature(nx, ny)
        inputs[9] = world.genetic_sim_to(self, ahead)
        # 10: last move x  (−1→0, 0→0.5, +1→1)
        inputs[10] = (DIRS[self.last_dir][0] + 1) / 2.0
        # 11: last move y
        inputs[11] = (DIRS[self.last_dir][1] + 1) / 2.0
        # 12: forward cell blocked
        inputs[12] = 1.0 if (ahead is not None) else 0.0
        # 13: constant bias
        inputs[13] = 1.0

        return inputs

    # ──────────────────────────────────────────────────────────────────────────

    def step(self, world, sim_step: int):
        """Execute one simulation step: sense → think → act."""
        if not self.alive:
            return

        self.age += 1
        inputs  = self._sense(world, sim_step)
        actions = self.brain.forward(inputs)

        # Choose the action with the highest activation magnitude
        # (ties broken by random, threshold applied so weak signals = noop)
        THRESHOLD = 0.1
        best_act = None
        best_val = THRESHOLD
        for i, v in enumerate(actions):
            if abs(v) > best_val:
                best_val = abs(v)
                best_act = i

        if best_act is not None:
            self._execute_action(best_act, actions[best_act], world)

    # ──────────────────────────────────────────────────────────────────────────

    def _execute_action(self, action_id: int, strength: float, world):
        """Carry out a chosen action."""
        rng = world.rng

        if action_id == 0:   # move_x  (positive = east, negative = west)
            dx = 1 if strength > 0 else -1
            world.move_creature(self, dx, 0)

        elif action_id == 1: # move_y  (positive = north, negative = south)
            dy = 1 if strength > 0 else -1
            world.move_creature(self, 0, dy)

        elif action_id == 2: # move_random
            d  = int(rng.integers(0, 8))
            dx, dy = DIRS[d]
            if world.move_creature(self, dx, dy):
                self.last_dir = d

        elif action_id == 3: # move_forward (continue last dir)
            dx, dy = DIRS[self.last_dir]
            world.move_creature(self, dx, dy)

        elif action_id == 4: # turn_left (−45°) then move
            self.last_dir = (self.last_dir - 1) % 8
            dx, dy = DIRS[self.last_dir]
            world.move_creature(self, dx, dy)

        elif action_id == 5: # turn_right (+45°) then move
            self.last_dir = (self.last_dir + 1) % 8
            dx, dy = DIRS[self.last_dir]
            world.move_creature(self, dx, dy)

        elif action_id == 6: # reverse
            self.last_dir = (self.last_dir + 4) % 8
            dx, dy = DIRS[self.last_dir]
            world.move_creature(self, dx, dy)

        elif action_id == 7: # noop or kill
            if KILL_ENABLED:
                dx, dy = DIRS[self.last_dir]
                world.kill_creature_at(self.x + dx, self.y + dy)
