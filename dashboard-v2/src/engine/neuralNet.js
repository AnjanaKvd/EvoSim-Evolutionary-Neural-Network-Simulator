/* ============================================================
   NEURAL NETWORK MODULE
   ============================================================ */

import { decodeGene } from './genome';

export const NUM_SENSORS = 14;
export const NUM_ACTIONS = 8;
export const SENSOR_LABELS = ["loc_x", "loc_y", "age", "random", "oscillator", "bdist_x", "bdist_y", "pop_density", "pop_grad_fwd", "genetic_sim_fwd", "last_move_x", "last_move_y", "fwd_blocked", "constant"];
export const ACTION_LABELS = ["move_x", "move_y", "move_random", "move_forward", "turn_left", "turn_right", "reverse", "noop/kill"];
export const DIRS = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];

export class NeuralNet {
    constructor(genome, maxInternal) {
        this.maxInternal = maxInternal;
        this.conns = this._parse(genome);
        this.conns = this._prune(this.conns);
    }
    _parse(genome) {
        return genome.map(g => decodeGene(g, NUM_SENSORS, NUM_ACTIONS, this.maxInternal));
    }
    _prune(conns) {
        const useful = new Set();
        let changed = true;
        while (changed) {
            changed = false;
            for (const c of conns) {
                if (c.sinkType === 1) {
                    if (c.sourceType === 1 && !useful.has(c.sourceId)) { useful.add(c.sourceId); changed = true; }
                } else if (c.sinkType === 0 && useful.has(c.sinkId)) {
                    if (c.sourceType === 1 && !useful.has(c.sourceId)) { useful.add(c.sourceId); changed = true; }
                }
            }
        }
        return conns.filter(c => c.sinkType === 1 || useful.has(c.sinkId));
    }
    forward(sensors) {
        const internal = new Float32Array(this.maxInternal);
        for (let pass = 0; pass < 2; pass++) {
            const next = new Float32Array(this.maxInternal);
            for (const c of this.conns) {
                if (c.sinkType === 0) {
                    const src = c.sourceType === 0 ? sensors[c.sourceId] : internal[c.sourceId];
                    next[c.sinkId] += src * c.weight;
                }
            }
            for (let i = 0; i < this.maxInternal; i++) internal[i] = Math.tanh(next[i]);
        }
        const actions = new Float32Array(NUM_ACTIONS);
        for (const c of this.conns) {
            if (c.sinkType === 1) {
                const src = c.sourceType === 0 ? sensors[c.sourceId] : internal[c.sourceId];
                actions[c.sinkId] += src * c.weight;
            }
        }
        for (let i = 0; i < NUM_ACTIONS; i++) actions[i] = Math.tanh(actions[i]);
        return actions;
    }
}
