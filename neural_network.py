"""
Neural Network Brain for EvoSim.

Each gene specifies one synaptic connection:
  sensor/internal → internal/action

Forward pass (per simulation step):
  1. Read sensory inputs (floats 0..1)
  2. Propagate through internal neurons (tanh activation, 2 iterations)
  3. Fire action neurons (tanh output → probability / magnitude)
"""

import numpy as np
from genome import decode_gene
from config import NUM_SENSORS, NUM_ACTIONS, MAX_INTERNAL_NEURONS


class NeuralNetwork:
    """
    Tiny neural network built entirely from a creature's genome.
    Topology: sensors → [internals] → actions
    """

    def __init__(self, genome: list):
        self.genome     = genome
        self.n_sensors  = NUM_SENSORS
        self.n_actions  = NUM_ACTIONS
        self.n_internal = MAX_INTERNAL_NEURONS

        # Parse all genes into connection lists
        self._connections = self._parse_genome(genome)

        # Prune dead-end internal neurons (inputs but no output path)
        self._connections = self._prune_dead_ends(self._connections)

        # State vectors (updated each forward pass)
        self._internal_vals = np.zeros(self.n_internal, dtype=np.float32)
        self._action_vals   = np.zeros(self.n_actions,  dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────

    def _parse_genome(self, genome: list) -> list:
        """Decode each gene into a connection dict."""
        connections = []
        for raw_gene in genome:
            c = decode_gene(int(raw_gene))
            connections.append(c)
        return connections

    def _prune_dead_ends(self, connections: list) -> list:
        """
        Remove connections whose sink is an internal neuron that has no
        outgoing path to an action neuron (they'd never affect behaviour).
        """
        # Find internal neurons that feed at least one action (directly or indirectly)
        # Iterative: start from internals that directly connect to actions,
        # then expand backward.
        useful = set()
        changed = True
        while changed:
            changed = False
            for c in connections:
                if c["sink_type"] == 1:                         # → action
                    # source internal (if any) is useful
                    if c["source_type"] == 1 and c["source_id"] not in useful:
                        useful.add(c["source_id"])
                        changed = True
                elif c["sink_type"] == 0 and c["sink_id"] in useful:
                    # this internal feeds a useful internal
                    if c["source_type"] == 1 and c["source_id"] not in useful:
                        useful.add(c["source_id"])
                        changed = True

        pruned = []
        for c in connections:
            if c["sink_type"] == 1:
                pruned.append(c)        # always keep → action connections
            elif c["sink_id"] in useful:
                pruned.append(c)        # keep → internal only if useful
        return pruned

    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, sensor_inputs: np.ndarray) -> np.ndarray:
        """
        Run one forward pass.

        Args:
            sensor_inputs: float32 array of shape (NUM_SENSORS,), values 0..1

        Returns:
            action_vals: float32 array of shape (NUM_ACTIONS,), values −1..1
        """
        self._internal_vals[:] = 0.0
        self._action_vals[:]   = 0.0

        # Accumulate weighted inputs into internal neurons first (2 passes
        # lets signals propagate through multi-hop internal chains).
        for _ in range(2):
            new_internal = np.zeros(self.n_internal, dtype=np.float32)
            for c in self._connections:
                if c["sink_type"] == 0:   # → internal
                    src = (sensor_inputs[c["source_id"]]
                           if c["source_type"] == 0
                           else self._internal_vals[c["source_id"]])
                    new_internal[c["sink_id"]] += src * c["weight"]
            self._internal_vals = np.tanh(new_internal)

        # Accumulate into action neurons
        action_sums = np.zeros(self.n_actions, dtype=np.float32)
        for c in self._connections:
            if c["sink_type"] == 1:   # → action
                src = (sensor_inputs[c["source_id"]]
                       if c["source_type"] == 0
                       else self._internal_vals[c["source_id"]])
                action_sums[c["sink_id"]] += src * c["weight"]

        self._action_vals = np.tanh(action_sums)
        return self._action_vals.copy()

    # ──────────────────────────────────────────────────────────────────────────

    def get_active_connections(self) -> list:
        """Return the pruned connection list (useful for visualisation)."""
        return self._connections

    def summary(self) -> str:
        lines = [f"NeuralNetwork ({len(self._connections)} active connections)"]
        for c in self._connections:
            src_label = ("S" if c["source_type"] == 0 else "I")
            snk_label = ("I" if c["sink_type"]   == 0 else "A")
            lines.append(
                f"  {src_label}{c['source_id']:02d} → {snk_label}{c['sink_id']:02d}"
                f"  w={c['weight']:+.3f}"
            )
        return "\n".join(lines)
