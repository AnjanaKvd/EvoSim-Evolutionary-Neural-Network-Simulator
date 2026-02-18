"""
EvoSim Configuration
All tunable parameters for the evolutionary neural network simulation.
"""

# ─── World ────────────────────────────────────────────────────────────────────
WORLD_WIDTH  = 128   # grid cells east-west
WORLD_HEIGHT = 128   # grid cells north-south

# ─── Population ───────────────────────────────────────────────────────────────
POPULATION       = 1000   # creatures per generation
MAX_GENERATIONS  = 1200   # how many generations to run
STEPS_PER_GEN    = 300    # simulator steps each creature lives

# ─── Genome / Brain ───────────────────────────────────────────────────────────
GENOME_SIZE         = 16    # number of genes (= max neural connections)
MAX_INTERNAL_NEURONS = 4    # available hidden neurons
MUTATION_RATE       = 0.001 # probability a single bit flips during copy
WEIGHT_DIVISOR      = 8000  # divides raw int16 weight → small float

# ─── Neural Network ───────────────────────────────────────────────────────────
# Sensory inputs available to every creature (index → meaning)
SENSOR_LABELS = {
    0:  "loc_x",              # x-position  (0→1)
    1:  "loc_y",              # y-position  (0→1)
    2:  "age",                # age fraction of lifetime (0→1)
    3:  "random",             # pure noise  (0→1 each step)
    4:  "oscillator",         # sin oscillator
    5:  "bdist_x",            # distance to nearest east/west wall (0→1)
    6:  "bdist_y",            # distance to nearest north/south wall (0→1)
    7:  "pop_density",        # nearby population density (0→1)
    8:  "pop_grad_fwd",       # population gradient in forward direction
    9:  "genetic_sim_fwd",    # genome similarity to creature directly ahead
    10: "last_move_x",        # last x displacement (−1,0,+1 → 0,0.5,1)
    11: "last_move_y",        # last y displacement
    12: "fwd_blocked",        # 1 if cell directly ahead is occupied
    13: "constant",           # always 1.0 (bias-like neuron)
}
NUM_SENSORS = len(SENSOR_LABELS)

# Action outputs available to every creature (index → meaning)
ACTION_LABELS = {
    0: "move_x",        # move east (+) / west (−)
    1: "move_y",        # move north (+) / south (−)
    2: "move_random",   # move to a random adjacent cell
    3: "move_forward",  # continue last direction
    4: "turn_left",     # rotate 90° CCW then move forward
    5: "turn_right",    # rotate 90° CW  then move forward
    6: "reverse",       # flip direction then move forward
    7: "noop",          # no operation (waste a turn)
}
NUM_ACTIONS = len(ACTION_LABELS)

# ─── Selection Criteria ───────────────────────────────────────────────────────
# Built-in modes:
#   "east"        – survive if x >= WORLD_WIDTH/2
#   "west"        – survive if x <  WORLD_WIDTH/2
#   "west_east"   – survive if in left OR right strip
#   "corners"     – survive if in any corner quadrant
#   "center"      – survive if within center_radius of centre
#   "radioactive" – survive based on wall avoidance (radioactivity challenge)

SELECTION_MODE   = "east"
CENTER_RADIUS    = 20       # used by "center" mode
STRIP_WIDTH      = 32       # used by "west_east" mode (each side)
CORNER_SIZE      = 32       # used by "corners" mode

# ─── Radioactive Challenge ─────────────────────────────────────────────────────
# First half of each generation: west wall is radioactive.
# Second half: east wall is radioactive.
# Creatures accumulate dose; die if dose exceeds threshold.
RADIO_MAX_DOSE   = 1.0      # creature dies above this
RADIO_FALLOFF    = 0.04     # exponential falloff constant

# ─── Kill Neuron (optional) ────────────────────────────────────────────────────
KILL_ENABLED     = False    # if True, action index 7 becomes kill instead of noop

# ─── Output / Logging ─────────────────────────────────────────────────────────
SAVE_DIR           = "output"      # directory for saved images and charts
SNAPSHOT_INTERVAL  = 50            # save a world snapshot every N generations
SAVE_NEURAL_SAMPLE = True          # save neural-network diagrams
LOG_CSV            = True          # write per-generation CSV log
