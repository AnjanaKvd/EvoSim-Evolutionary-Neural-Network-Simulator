# EvoSim — Evolutionary Neural Network Simulator

A Python implementation of the evolutionary simulation from Dave Miller's video. Creatures live in a 2-D grid world, carry genomes, build neural network brains from those genomes, and evolve behaviour through natural selection.

---

## Key Concepts (from the video)

| Requirement | Implementation |
|---|---|
| Self-replicating entities | Creatures with genomes that copy to offspring |
| Blueprint (genome) | List of 32-bit genes encoding neural connections |
| Inherited blueprint | Crossover + copy from parent pair |
| Mutations | Bit-flip mutation per gene at configurable rate |
| Natural selection | Spatial survival criteria (east half, corners, etc.) |

---

## Project Structure

```
evosim/
├── config.py           ← All tunable parameters
├── genome.py           ← Genome encoding, mutation, crossover
├── neural_network.py   ← Brain built from genome genes
├── creature.py         ← Agent: sense → think → act
├── world.py            ← 2-D grid, movement, sensory queries
├── simulation.py       ← Evolution loop engine
├── visualizer.py       ← Snapshots, charts, neural diagrams
├── main.py             ← CLI entry point
├── fast_demo.py        ← Quick 100-gen demo
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ with `numpy`, `matplotlib`, `Pillow`.

---

## Running

### Quick demo (100 generations)
```bash
python fast_demo.py
```

### Full simulation with CLI
```bash
# Default: east-half survival
python main.py

# Choose a scenario
python main.py --scenario west_east
python main.py --scenario corners
python main.py --scenario center
python main.py --scenario radioactive
python main.py --scenario kill       # enable kill neuron experiment

# Custom parameters
python main.py --gens 2000 --pop 1000 --brain_size 24 --mutation 0.001

# Turn off mutations (demonstrate their importance)
python main.py --no_mutation

# Reproducible run
python main.py --seed 42
```

---

## Scenarios

| Scenario | Description | What evolves |
|---|---|---|
| `east` | Survive in right half | Migrate east |
| `west` | Survive in left half | Migrate west |
| `west_east` | Survive in either side strip | Split east/west instinct |
| `corners` | Survive in any corner | Navigate to a corner |
| `center` | Survive inside centre circle | Converge on centre |
| `radioactive` | Dodge walls that alternate radiating | Learn to cross midline |
| `kill` | Kill neuron enabled | Bi-stable violent/peaceful society |

---

## How the Genome Works

Each gene is a 32-bit integer encoding one synaptic connection:

```
Bit 31      : source type   (0 = sensor, 1 = internal neuron)
Bits 30–24  : source index  (7 bits)
Bit 23      : sink type     (0 = internal, 1 = action neuron)
Bits 22–16  : sink index    (7 bits)
Bits 15–0   : weight        (signed int16 / WEIGHT_DIVISOR → small float)
```

A genome of size N creates up to N connections.  Dead-end internal neurons (those with no path to any action neuron) are pruned before computation.

---

## Sensory Inputs (14)

| # | Name | Description |
|---|---|---|
| 0 | loc_x | Normalised X position |
| 1 | loc_y | Normalised Y position |
| 2 | age | Fraction of lifetime elapsed |
| 3 | random | Pure noise each step |
| 4 | oscillator | ~30-step sine wave |
| 5 | bdist_x | Distance to nearest E/W wall |
| 6 | bdist_y | Distance to nearest N/S wall |
| 7 | pop_density | Local population density |
| 8 | pop_grad_fwd | Population gradient ahead |
| 9 | genetic_sim_fwd | Genome similarity to creature directly ahead |
| 10 | last_move_x | Last x displacement |
| 11 | last_move_y | Last y displacement |
| 12 | fwd_blocked | Cell directly ahead is occupied |
| 13 | constant | Always 1.0 (bias-like) |

## Action Outputs (8)

| # | Name | Description |
|---|---|---|
| 0 | move_x | Move east (+) or west (−) |
| 1 | move_y | Move north (+) or south (−) |
| 2 | move_random | Random adjacent step |
| 3 | move_forward | Continue in last direction |
| 4 | turn_left | Rotate 90° CCW then step |
| 5 | turn_right | Rotate 90° CW then step |
| 6 | reverse | Flip direction then step |
| 7 | noop / kill | No-op, or kill neighbour if KILL_ENABLED |

---

## Outputs

All outputs are saved to `output/<scenario>/`:

```
output/
└── east/
    ├── snapshots/          ← gen_000000.png, gen_000050.png …
    ├── charts/             ← evolution_final.png
    ├── neural/             ← gen_000050_best_survivor.png
    └── evolution_log.csv   ← per-generation stats
```

### Evolution chart
- **Green** line: number of survivors per generation  
- **Purple** dashed: genetic diversity (0 = identical, 1 = maximally diverse)  
- **Orange** (if kill enabled): murders per generation

### Neural network diagram
- **Blue** nodes: sensory input neurons  
- **Grey** nodes: internal (hidden) neurons  
- **Pink** nodes: action output neurons  
- **Green** edges: positive (excitatory) weights  
- **Red** edges: negative (inhibitory) weights  

---

## Key Parameters (config.py)

| Parameter | Default | Description |
|---|---|---|
| `WORLD_WIDTH/HEIGHT` | 128×128 | Grid dimensions |
| `POPULATION` | 1000 | Creatures per generation |
| `GENOME_SIZE` | 16 | Genes = max neural connections |
| `MAX_INTERNAL_NEURONS` | 4 | Hidden neurons available |
| `MUTATION_RATE` | 0.001 | Bit-flip probability per bit |
| `STEPS_PER_GEN` | 300 | Simulator steps per lifetime |
| `KILL_ENABLED` | False | Enable kill neuron |

---

## What You'll Observe

1. **Generation 0**: random behaviour, ~50% survival (luck)
2. **Early generations**: rapid improvement as directional instinct spreads
3. **Mid generations**: near-optimal behaviour, diversity narrows
4. **After environmental change** (barriers/radioactivity): temporary drop, then recovery via mutation
5. **No-mutation run**: permanent stagnation after environmental change

> *"Whatever reproduces, reproduces. Whatever doesn't, doesn't."*  
> — Dave Miller
