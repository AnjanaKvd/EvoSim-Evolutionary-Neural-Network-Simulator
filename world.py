"""
World Grid for EvoSim.

The world is a fixed-size 2-D grid. Each cell can hold at most one
creature. The world also exposes helper methods that creatures use
for sensing their environment.
"""

import numpy as np
from genome import genome_similarity
from config import WORLD_WIDTH, WORLD_HEIGHT


class World:
    """
    Manages the spatial grid and all creature movement.
    """

    def __init__(self, width: int = WORLD_WIDTH, height: int = WORLD_HEIGHT,
                 seed: int = None):
        self.width  = width
        self.height = height
        self.rng    = np.random.default_rng(seed)
        # grid[y][x] = Creature or None
        self._grid  = [[None] * width for _ in range(height)]
        self.creatures = []      # all living creatures this generation
        self.murder_count = 0    # reset each generation

    # ──────────────────────────────────────────────────────────────────────────
    # Placement helpers
    # ──────────────────────────────────────────────────────────────────────────

    def place_creature(self, creature) -> bool:
        """Put a creature at its (x,y) if the cell is free."""
        if self._in_bounds(creature.x, creature.y):
            if self._grid[creature.y][creature.x] is None:
                self._grid[creature.y][creature.x] = creature
                return True
        return False

    def remove_creature(self, creature):
        """Remove creature from grid."""
        if self._in_bounds(creature.x, creature.y):
            if self._grid[creature.y][creature.x] is creature:
                self._grid[creature.y][creature.x] = None

    def clear(self):
        """Remove all creatures from the grid."""
        self._grid = [[None] * self.width for _ in range(self.height)]
        self.creatures = []
        self.murder_count = 0

    def populate(self, creatures: list):
        """
        Place a list of creatures at random empty cells.
        Assigns (x,y) positions before placement.
        """
        self.clear()
        # Generate a shuffled list of all positions
        positions = [(x, y)
                     for y in range(self.height)
                     for x in range(self.width)]
        positions = [positions[i] for i in self.rng.permutation(len(positions))]
        placed = 0
        for creature in creatures:
            if placed >= len(positions):
                break
            creature.x, creature.y = positions[placed]
            self._grid[creature.y][creature.x] = creature
            placed += 1
        self.creatures = list(creatures[:placed])

    # ──────────────────────────────────────────────────────────────────────────
    # Movement
    # ──────────────────────────────────────────────────────────────────────────

    def move_creature(self, creature, dx: int, dy: int) -> bool:
        """
        Attempt to move creature by (dx, dy).
        Returns True on success, False if blocked or OOB.
        Movement is clamped at world boundaries (walls).
        """
        nx = max(0, min(self.width  - 1, creature.x + dx))
        ny = max(0, min(self.height - 1, creature.y + dy))
        if nx == creature.x and ny == creature.y:
            return False     # wall block
        if self._grid[ny][nx] is not None:
            return False     # occupied
        # Move
        self._grid[creature.y][creature.x] = None
        creature.x, creature.y = nx, ny
        self._grid[ny][nx] = creature
        # Update last direction
        if dx != 0 or dy != 0:
            from creature import DIRS
            for i, (ddx, ddy) in enumerate(DIRS):
                if ddx == (1 if dx > 0 else (-1 if dx < 0 else 0)) and \
                   ddy == (1 if dy > 0 else (-1 if dy < 0 else 0)):
                    creature.last_dir = i
                    break
        return True

    def kill_creature_at(self, x: int, y: int):
        """Kill creature at grid cell (x,y) if present."""
        if not self._in_bounds(x, y):
            return
        victim = self._grid[y][x]
        if victim and victim.alive:
            victim.alive = False
            self._grid[y][x] = None
            self.murder_count += 1

    # ──────────────────────────────────────────────────────────────────────────
    # Sensing helpers (used by creatures)
    # ──────────────────────────────────────────────────────────────────────────

    def get_creature(self, x: int, y: int):
        """Return creature at (x,y) or None."""
        if self._in_bounds(x, y):
            return self._grid[y][x]
        return None

    def local_density(self, cx: int, cy: int, radius: int = 2) -> int:
        """Count living creatures within a square of given radius (excl. self)."""
        count = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if self._in_bounds(nx, ny) and self._grid[ny][nx] is not None:
                    count += 1
        return count

    def forward_density_gradient(self, cx: int, cy: int, dir_idx: int) -> float:
        """
        Measure population density in a cone of 5 cells ahead minus 5 behind.
        Returns a value roughly in [−1, 1].
        """
        from creature import DIRS
        dx, dy = DIRS[dir_idx]
        fwd = sum(
            1 for step in range(1, 6)
            if self._in_bounds(cx + dx*step, cy + dy*step)
            and self._grid[cy + dy*step][cx + dx*step] is not None
        )
        bwd = sum(
            1 for step in range(1, 6)
            if self._in_bounds(cx - dx*step, cy - dy*step)
            and self._grid[cy - dy*step][cx - dx*step] is not None
        )
        return (fwd - bwd) / 5.0

    def genetic_sim_to(self, creature, other) -> float:
        """Genetic similarity between creature and other (0 if no other)."""
        if other is None:
            return 0.0
        return genome_similarity(creature.genome, other.genome)

    # ──────────────────────────────────────────────────────────────────────────
    # Snapshot for visualisation
    # ──────────────────────────────────────────────────────────────────────────

    def snapshot(self):
        """
        Returns two arrays for visualisation:
          positions: list of (x, y) for every living creature
          colors:    list of (r, g, b) tuples
        """
        positions = []
        colors    = []
        for c in self.creatures:
            if c.alive:
                positions.append((c.x, c.y))
                colors.append(c.color)
        return positions, colors

    # ──────────────────────────────────────────────────────────────────────────

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height
