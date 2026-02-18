"""
Visualizer for EvoSim.

Produces:
  1. World snapshots  – coloured scatter plot of creature positions
  2. Evolution chart  – survivors + diversity + murders over generations
  3. Neural network diagrams – wiring of a sampled creature's brain
  4. CSV log          – per-generation stats
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb

from config import (
    WORLD_WIDTH, WORLD_HEIGHT, SAVE_DIR, LOG_CSV,
    NUM_SENSORS, NUM_ACTIONS, SENSOR_LABELS, ACTION_LABELS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Directory setup
# ──────────────────────────────────────────────────────────────────────────────

def ensure_dirs(base: str = SAVE_DIR):
    for sub in ("snapshots", "charts", "neural"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# World snapshot
# ──────────────────────────────────────────────────────────────────────────────

def save_world_snapshot(world, generation: int, survivors: list,
                        selection_mode: str, base: str = SAVE_DIR):
    """
    Render the current world as a scatter plot.
    Survivors are highlighted with a white ring.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(-1, world.width)
    ax.set_ylim(-1, world.height)
    ax.set_aspect("equal")
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")
    ax.set_title(f"Generation {generation}  "
                 f"({len(survivors)}/{len(world.creatures)} survived)",
                 color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    positions, colors = world.snapshot()
    if positions:
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        rgba = [[r/255, g/255, b/255, 1.0] for (r, g, b) in colors]
        ax.scatter(xs, ys, c=rgba, s=4, linewidths=0)

    # Draw selection zones
    _draw_selection_zone(ax, selection_mode, world.width, world.height)

    path = os.path.join(base, "snapshots", f"gen_{generation:06d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _draw_selection_zone(ax, mode: str, W: int, H: int):
    """Overlay the safe/spawn zone in transparent green."""
    from config import STRIP_WIDTH, CORNER_SIZE, CENTER_RADIUS
    alpha = 0.15
    color = "lime"

    if mode == "east":
        ax.axvspan(W // 2, W, alpha=alpha, color=color)

    elif mode == "west":
        ax.axvspan(0, W // 2, alpha=alpha, color=color)

    elif mode == "west_east":
        ax.axvspan(0, STRIP_WIDTH, alpha=alpha, color=color)
        ax.axvspan(W - STRIP_WIDTH, W, alpha=alpha, color=color)

    elif mode == "corners":
        for rx, ry in [(0, 0), (W-CORNER_SIZE, 0),
                       (0, H-CORNER_SIZE), (W-CORNER_SIZE, H-CORNER_SIZE)]:
            rect = mpatches.Rectangle(
                (rx, ry), CORNER_SIZE, CORNER_SIZE,
                linewidth=0, edgecolor=None, facecolor=color, alpha=alpha)
            ax.add_patch(rect)

    elif mode == "center":
        circle = mpatches.Circle(
            (W//2, H//2), CENTER_RADIUS,
            color=color, alpha=alpha)
        ax.add_patch(circle)

    elif mode == "radioactive":
        # Show radioactive walls in red
        ax.axvspan(0, W * 0.1, alpha=0.1, color="red")
        ax.axvspan(W * 0.9, W, alpha=0.1, color="red")


# ──────────────────────────────────────────────────────────────────────────────
# Evolution statistics chart
# ──────────────────────────────────────────────────────────────────────────────

def save_evolution_chart(stats: list, base: str = SAVE_DIR,
                         filename: str = "evolution.png"):
    """
    Plot survivors, genetic diversity, and murders across all generations.
    Mirrors the green/purple/orange chart from the video.
    """
    if not stats:
        return
    gens      = [s["generation"]   for s in stats]
    survivors = [s["survivors"]    for s in stats]
    diversity = [s["diversity"]    for s in stats]
    murdered  = [s["murdered"]     for s in stats]
    popul     = [s["population"]   for s in stats]

    fig, ax1 = plt.subplots(figsize=(12, 5), dpi=100)
    fig.patch.set_facecolor("#111111")
    ax1.set_facecolor("#111111")

    # Survivors (green, left axis 0–population)
    ax1.plot(gens, survivors, color="#44FF44", linewidth=1.2,
             label="Survivors", zorder=3)
    ax1.set_ylabel("Count", color="white")
    ax1.set_ylim(0, max(popul) * 1.05 if popul else 1)
    ax1.tick_params(axis="both", colors="white")
    ax1.set_xlabel("Generation", color="white")

    ax2 = ax1.twinx()
    ax2.set_facecolor("#111111")

    # Genetic diversity (purple, right axis 0–1)
    ax2.plot(gens, diversity, color="#CC44FF", linewidth=1.0,
             linestyle="--", label="Diversity", zorder=2)

    # Murders (orange, also left axis but secondary)
    if any(m > 0 for m in murdered):
        ax1.plot(gens, murdered, color="#FF8800", linewidth=1.0,
                 alpha=0.8, label="Murders", zorder=2)

    ax2.set_ylabel("Genetic diversity (0–1)", color="white")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(colors="white")

    for spine in ax1.spines.values():
        spine.set_edgecolor("#444444")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor="#222222", labelcolor="white",
               loc="lower right", fontsize=8)

    ax1.set_title("Evolutionary Progress", color="white", fontsize=12)
    plt.tight_layout()
    path = os.path.join(base, "charts", filename)
    plt.savefig(path, dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Neural network diagram
# ──────────────────────────────────────────────────────────────────────────────

def save_neural_diagram(creature, generation: int, label: str = "",
                        base: str = SAVE_DIR):
    """
    Draw the creature's neural-network wiring as a layered graph.
    Sensors (blue) → internals (grey) → actions (pink).
    Green edges = positive weights, red edges = negative.
    """
    connections = creature.brain.get_active_connections()
    if not connections:
        return

    # Collect active nodes
    active_sensors  = sorted(set(
        c["source_id"] for c in connections if c["source_type"] == 0))
    active_internals = sorted(set(
        c["source_id"] for c in connections if c["source_type"] == 1
    ) | set(
        c["sink_id"]   for c in connections if c["sink_type"]   == 0))
    active_actions   = sorted(set(
        c["sink_id"]   for c in connections if c["sink_type"]   == 1))

    # Assign (x, y) positions for each node
    node_pos = {}

    def _y_positions(ids, n_total):
        if not ids:
            return {}
        step = 1.0 / (len(ids) + 1)
        return {nid: (i + 1) * step for i, nid in enumerate(sorted(ids))}

    for nid in active_sensors:
        ys = list(range(len(active_sensors)))
        idx = active_sensors.index(nid)
        node_pos[("S", nid)] = (0.0, (idx + 1) / (len(active_sensors) + 1))

    for nid in active_internals:
        idx = active_internals.index(nid)
        node_pos[("I", nid)] = (0.5, (idx + 1) / (len(active_internals) + 1))

    for nid in active_actions:
        idx = active_actions.index(nid)
        node_pos[("A", nid)] = (1.0, (idx + 1) / (len(active_actions) + 1))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    ax.axis("off")
    ax.set_xlim(-0.15, 1.35)
    ax.set_ylim(-0.05, 1.05)

    # Draw edges
    for c in connections:
        src_key = ("S" if c["source_type"] == 0 else "I", c["source_id"])
        snk_key = ("I" if c["sink_type"]   == 0 else "A", c["sink_id"])
        if src_key not in node_pos or snk_key not in node_pos:
            continue
        x1, y1 = node_pos[src_key]
        x2, y2 = node_pos[snk_key]
        w       = c["weight"]
        color   = "#44FF44" if w >= 0 else "#FF4444"
        lw      = 0.5 + min(3.0, abs(w) * 2)
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=0.7, zorder=1)
        # Arrowhead midpoint
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.annotate("", xy=(x2, y2), xytext=(mx, my),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8),
                    zorder=2)

    # Draw nodes
    def _draw_nodes(keys, color, label_prefix, label_dict=None):
        for (kind, nid) in keys:
            if (kind, nid) not in node_pos:
                continue
            x, y = node_pos[(kind, nid)]
            circle = plt.Circle((x, y), 0.025, color=color, zorder=3)
            ax.add_patch(circle)
            lbl = label_dict.get(nid, f"{label_prefix}{nid}") if label_dict else f"{label_prefix}{nid}"
            ha  = "right" if kind == "S" else ("left" if kind == "A" else "center")
            offset_x = -0.03 if kind == "S" else (0.03 if kind == "A" else 0)
            ax.text(x + offset_x, y, lbl, color="white", fontsize=6.5,
                    ha=ha, va="center", zorder=4)

    _draw_nodes([("S", n) for n in active_sensors],   "#4499FF", "S", SENSOR_LABELS)
    _draw_nodes([("I", n) for n in active_internals], "#AAAAAA", "I")
    _draw_nodes([("A", n) for n in active_actions],   "#FF88AA", "A", ACTION_LABELS)

    # Column headers
    for tx, title in [(0.0, "Sensors"), (0.5, "Internals"), (1.0, "Actions")]:
        ax.text(tx, 1.03, title, color="#CCCCCC", ha="center",
                fontsize=9, fontweight="bold")

    ax.set_title(
        f"Gen {generation} — Brain of {label}  "
        f"({len(connections)} active connections)",
        color="white", fontsize=10, pad=4)

    path = os.path.join(base, "neural", f"gen_{generation:06d}_{label}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# CSV log
# ──────────────────────────────────────────────────────────────────────────────

def append_csv(stats: dict, base: str = SAVE_DIR):
    """Append one generation's stats to a CSV file."""
    if not LOG_CSV:
        return
    path = os.path.join(base, "evolution_log.csv")
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(stats)
