"""
EvoSim Server  –  Flask + Server-Sent Events
============================================

Endpoints:
  POST /start        Start (or restart) simulation with JSON config body
  POST /stop         Stop the running simulation
  POST /step         Run exactly one generation
  GET  /stream       SSE stream – browser subscribes here for live data
  GET  /status       Current sim state as JSON

Run:
  python server.py
  # → http://localhost:5000
"""

import threading
import queue
import json
import time
import sys
import os

from flask import Flask, Response, request, jsonify

# Make sure the evosim package is importable from this folder
sys.path.insert(0, os.path.dirname(__file__))

from simulation import Simulation
from config import (
    POPULATION, MAX_GENERATIONS, STEPS_PER_GEN,
    GENOME_SIZE, MUTATION_RATE, SELECTION_MODE,
    WORLD_WIDTH, WORLD_HEIGHT,
)

# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Global simulation state
_sim_thread:  threading.Thread | None = None
_stop_event   = threading.Event()
_gen_queue    = queue.Queue(maxsize=200)   # holds dicts to stream
_sim_status   = {
    "running":    False,
    "generation": 0,
    "max_gen":    0,
    "cfg":        {},
}
_status_lock  = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────────
# CORS helper – allow the React dev server (any origin) to call us
# ──────────────────────────────────────────────────────────────────────────────

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/", methods=["OPTIONS"])
@app.route("/<path:p>", methods=["OPTIONS"])
def preflight(p=""):
    return Response(status=200)


# ──────────────────────────────────────────────────────────────────────────────
# Simulation thread
# ──────────────────────────────────────────────────────────────────────────────

def _build_cfg(data: dict) -> dict:
    """Merge request JSON with defaults."""
    return {
        "world_width":    int(data.get("worldWidth",    WORLD_WIDTH)),
        "world_height":   int(data.get("worldHeight",   WORLD_HEIGHT)),
        "population":     int(data.get("population",    POPULATION)),
        "max_generations":int(data.get("maxGenerations",MAX_GENERATIONS)),
        "steps_per_gen":  int(data.get("stepsPerGen",   STEPS_PER_GEN)),
        "genome_size":    int(data.get("genomeSize",     GENOME_SIZE)),
        "max_internal":   int(data.get("maxInternal",   4)),
        "mutation_rate":  float(data.get("mutationRate", MUTATION_RATE)),
        "selection_mode": str(data.get("selectionMode", SELECTION_MODE)),
        "kill_enabled":   bool(data.get("killEnabled",  False)),
        "strip_width":    int(data.get("stripWidth",    32)),
        "corner_size":    int(data.get("cornerSize",    32)),
        "center_radius":  int(data.get("centerRadius",  20)),
    }


def _sim_worker(cfg: dict, stop_evt: threading.Event, out_q: queue.Queue):
    """Run full simulation in background thread; push each generation into queue."""

    def on_gen(gen_idx, stats, world, creatures, survivors):
        if stop_evt.is_set():
            return

        # Build compact creature snapshot (x, y, color)
        snapshot = [
            {"x": c.x, "y": c.y, "r": int(c.color[0]), "g": int(c.color[1]), "b": int(c.color[2])}
            for c in creatures if c.alive
        ]

        # Best survivor genome (top 1 by connection count)
        best_genome   = []
        best_conns    = []
        if survivors:
            best = max(survivors, key=lambda c: len(c.brain.get_active_connections()))
            best_genome = [int(g) & 0xFFFFFFFF for g in best.genome]
            best_conns  = [
                {
                    "sourceType": int(c["source_type"]),
                    "sourceId":   int(c["source_id"]),
                    "sinkType":   int(c["sink_type"]),
                    "sinkId":     int(c["sink_id"]),
                    "weight":     round(float(c["weight"]), 4),
                }
                for c in best.brain.get_active_connections()
            ]

        payload = {
            "type":        "generation",
            "gen":         gen_idx,
            "maxGen":      cfg["max_generations"],
            "survivors":   stats["survivors"],
            "population":  stats["population"],
            "survivalPct": round(stats["survival_pct"], 1),
            "diversity":   round(stats["diversity"], 4),
            "murders":     stats["murdered"],
            "snapshot":    snapshot,
            "bestGenome":  best_genome,
            "bestConns":   best_conns,
        }

        with _status_lock:
            _sim_status["generation"] = gen_idx

        # Non-blocking put; drop oldest frame if queue full
        if out_q.full():
            try:
                out_q.get_nowait()
            except queue.Empty:
                pass
        out_q.put(payload)

    # Patch config module so Simulation picks up new values
    import config as conf_mod
    import config as _conf
    _conf.KILL_ENABLED = cfg["kill_enabled"]

    sim = Simulation(
        population      = cfg["population"],
        max_generations = cfg["max_generations"],
        steps_per_gen   = cfg["steps_per_gen"],
        genome_size     = cfg["genome_size"],
        mutation_rate   = cfg["mutation_rate"],
        selection_mode  = cfg["selection_mode"],
        world_width     = cfg["world_width"],
        world_height    = cfg["world_height"],
        seed            = None,
        on_gen_callback = on_gen,
    )

    # Override config module fields needed inside Simulation
    import config as cmod
    cmod.STRIP_WIDTH   = cfg["strip_width"]
    cmod.CORNER_SIZE   = cfg["corner_size"]
    cmod.CENTER_RADIUS = cfg["center_radius"]

    with _status_lock:
        _sim_status["running"] = True

    try:
        while not stop_evt.is_set():
            # Run one generation manually so we can check stop_evt between gens
            if sim.generation >= cfg["max_generations"]:
                break
            from genome import random_genome
            if sim.generation == 0:
                genomes = [random_genome(cfg["genome_size"], sim.rng)
                           for _ in range(cfg["population"])]
                sim._run_all_generations_hooked(genomes, stop_evt)
                break
    finally:
        with _status_lock:
            _sim_status["running"] = False
        out_q.put({"type": "done", "gen": _sim_status["generation"]})


# ──────────────────────────────────────────────────────────────────────────────
# We need a small hook in Simulation so we can check stop_evt between gens
# ──────────────────────────────────────────────────────────────────────────────

def _patch_simulation():
    """Monkey-patch Simulation to add a stoppable run method."""
    from simulation import Simulation
    import time

    def _run_hooked(self, initial_genomes, stop_evt):
        genomes = initial_genomes
        for gen_idx in range(self.max_generations):
            if stop_evt.is_set():
                break
            self.generation = gen_idx
            t0 = time.time()
            survivors, stats = self._run_one_generation(genomes)
            stats["elapsed_s"] = round(time.time() - t0, 3)
            self.stats.append(stats)
            self._print_stats(gen_idx, stats)
            if self.on_gen_callback:
                self.on_gen_callback(gen_idx, stats, self.world,
                                     self.current_gen_creatures, survivors)
            if not survivors:
                break
            genomes = self._reproduce(survivors)

    Simulation._run_all_generations_hooked = _run_hooked

_patch_simulation()


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/start", methods=["POST"])
def start():
    global _sim_thread, _stop_event, _gen_queue

    # Stop any running sim
    _stop_event.set()
    if _sim_thread and _sim_thread.is_alive():
        _sim_thread.join(timeout=3)

    # Reset
    _stop_event = threading.Event()
    _gen_queue  = queue.Queue(maxsize=200)
    with _status_lock:
        _sim_status["generation"] = 0
        _sim_status["running"]    = False

    cfg = _build_cfg(request.get_json(force=True) or {})
    with _status_lock:
        _sim_status["cfg"]    = cfg
        _sim_status["max_gen"]= cfg["max_generations"]

    _sim_thread = threading.Thread(
        target=_sim_worker,
        args=(cfg, _stop_event, _gen_queue),
        daemon=True,
    )
    _sim_thread.start()
    return jsonify({"status": "started", "cfg": cfg})


@app.route("/stop", methods=["POST"])
def stop():
    _stop_event.set()
    return jsonify({"status": "stopped"})


@app.route("/step", methods=["POST"])
def step_one():
    """Run exactly one generation (convenience for manual stepping)."""
    return jsonify({"status": "not_implemented",
                    "hint": "Use /start with maxGenerations=1 instead"})


@app.route("/status", methods=["GET"])
def status():
    with _status_lock:
        return jsonify(dict(_sim_status))


@app.route("/stream", methods=["GET"])
def stream():
    """SSE endpoint – browser subscribes and receives each generation as an event."""

    def event_gen():
        # Send a hello so the browser knows it's connected
        yield "data: {\"type\": \"connected\"}\n\n"

        while True:
            try:
                payload = _gen_queue.get(timeout=1)
                yield f"data: {json.dumps(payload)}\n\n"
                if payload.get("type") == "done":
                    break
            except queue.Empty:
                # Keep-alive ping
                yield "data: {\"type\": \"ping\"}\n\n"

    return Response(
        event_gen(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",    # disable nginx buffering if behind proxy
        },
    )


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  EvoSim Server  →  http://localhost:5000")
    print("  SSE stream     →  http://localhost:5000/stream")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
