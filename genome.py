"""
Genome encoding / decoding for EvoSim.

Each gene is a 32-bit integer divided into 5 fields:

 Bit 31     : source_type   (0=sensor neuron, 1=internal neuron)
 Bits 30-24 : source_id     (7 bits → index into sensor or internal pool)
 Bit  23    : sink_type     (0=internal neuron, 1=action neuron)
 Bits 22-16 : sink_id       (7 bits → index into internal or action pool)
 Bits 15-0  : weight        (signed int16, divided by WEIGHT_DIVISOR → float)
"""

import numpy as np
from config import (GENOME_SIZE, MUTATION_RATE, WEIGHT_DIVISOR,
                    NUM_SENSORS, NUM_ACTIONS, MAX_INTERNAL_NEURONS)

# ──────────────────────────────────────────────────────────────────────────────
# Gene helpers
# ──────────────────────────────────────────────────────────────────────────────

def decode_gene(gene: int) -> dict:
    """Unpack a 32-bit int into gene fields."""
    source_type = (gene >> 31) & 0x1
    source_id   = (gene >> 24) & 0x7F
    sink_type   = (gene >> 23) & 0x1
    sink_id     = (gene >> 16) & 0x7F
    # signed 16-bit weight
    raw_weight  = gene & 0xFFFF
    if raw_weight >= 0x8000:
        raw_weight -= 0x10000
    weight = raw_weight / WEIGHT_DIVISOR

    # Clamp indices to valid range
    if source_type == 0:
        source_id = source_id % NUM_SENSORS
    else:
        source_id = source_id % max(1, MAX_INTERNAL_NEURONS)

    if sink_type == 1:
        sink_id = sink_id % NUM_ACTIONS
    else:
        sink_id = sink_id % max(1, MAX_INTERNAL_NEURONS)

    return {
        "source_type": source_type,   # 0=sensor, 1=internal
        "source_id":   source_id,
        "sink_type":   sink_type,     # 0=internal, 1=action
        "sink_id":     sink_id,
        "weight":      weight,
    }


def encode_gene(source_type: int, source_id: int,
                sink_type: int,   sink_id: int,
                weight_float: float) -> int:
    """Pack gene fields back into a 32-bit int."""
    raw_weight = int(weight_float * WEIGHT_DIVISOR)
    raw_weight = max(-32768, min(32767, raw_weight))
    if raw_weight < 0:
        raw_weight += 0x10000
    gene  = (source_type & 0x1)  << 31
    gene |= (source_id   & 0x7F) << 24
    gene |= (sink_type   & 0x1)  << 23
    gene |= (sink_id     & 0x7F) << 16
    gene |= (raw_weight  & 0xFFFF)
    return gene


# ──────────────────────────────────────────────────────────────────────────────
# Population-level operations
# ──────────────────────────────────────────────────────────────────────────────

def random_genome(size: int = GENOME_SIZE, rng=None) -> list:
    """Generate a random genome as a list of 32-bit ints."""
    if rng is None:
        rng = np.random.default_rng()
    # Use uint32, interpret as int32 for signedness
    return list(rng.integers(0, 2**32, size=size, dtype=np.uint32).astype(np.int32))


def mutate_genome(genome: list, rate: float = MUTATION_RATE, rng=None) -> list:
    """
    Flip individual bits with probability `rate` per bit.
    Each 32-bit gene has 32 bits → expected flips ≈ rate * 32 per gene.
    """
    if rng is None:
        rng = np.random.default_rng()
    mutated = []
    for gene in genome:
        gene = int(gene) & 0xFFFFFFFF   # treat as uint32
        for bit in range(32):
            if rng.random() < rate:
                gene ^= (1 << bit)
        gene &= 0xFFFFFFFF
        # Reinterpret as signed int32
        if gene >= 0x80000000:
            gene -= 0x100000000
        mutated.append(int(gene))
    return mutated


def crossover(genome_a: list, genome_b: list, rng=None) -> list:
    """
    Single-point crossover: pick a random split point, take genes
    [0:split] from parent A and [split:] from parent B.
    """
    if rng is None:
        rng = np.random.default_rng()
    size = len(genome_a)
    split = int(rng.integers(0, size + 1))
    return list(genome_a[:split]) + list(genome_b[split:])


def genome_similarity(genome_a: list, genome_b: list) -> float:
    """
    Genetic similarity (0..1) based on fraction of identical bits.
    Used by the genetic-compatibility sensory neuron.
    """
    if not genome_a or not genome_b:
        return 0.0
    total_bits = 0
    matching   = 0
    for a, b in zip(genome_a, genome_b):
        xor = int(a) ^ int(b)
        matching   += 32 - bin(xor).count('1')
        total_bits += 32
    return matching / total_bits if total_bits else 1.0


def genome_to_color(genome: list) -> tuple:
    """
    Map a genome to an RGB colour so that genetically similar creatures
    have similar colours (useful visual diversity indicator).
    """
    if not genome:
        return (128, 128, 128)
    # XOR-fold all genes into 24 bits
    h = 0
    for g in genome:
        h ^= int(g) & 0xFFFFFF
    r = (h >> 16) & 0xFF
    g = (h >>  8) & 0xFF
    b =  h        & 0xFF
    # Brighten so they're visible
    r = max(50, r)
    g = max(50, g)
    b = max(50, b)
    return (r, g, b)
