/* ============================================================
   WORLD MODULE
   ============================================================ */

import { seededRng, shuffle } from './utils';
import { genomeSimilarity } from './genome';
import { NUM_SENSORS, NUM_ACTIONS, DIRS } from './neuralNet';

export class World {
    constructor(W, H) {
        this.W = W; this.H = H;
        this.grid = new Array(W * H).fill(null);
        this.creatures = [];
        this.murders = 0;
        this.rng = seededRng(Date.now());
    }
    idx(x, y) { return y * this.W + x; }
    inBounds(x, y) { return x >= 0 && x < this.W && y >= 0 && y < this.H; }
    get(x, y) { return this.inBounds(x, y) ? this.grid[this.idx(x, y)] : null; }
    set(x, y, c) { if (this.inBounds(x, y)) this.grid[this.idx(x, y)] = c; }
    clear() { this.grid.fill(null); this.creatures = []; this.murders = 0; }

    populate(creatures) {
        this.clear();
        const positions = shuffle(Array.from({ length: this.W * this.H }, (_, i) => i), this.rng);
        for (let i = 0; i < Math.min(creatures.length, positions.length); i++) {
            const pos = positions[i];
            const x = pos % this.W, y = Math.floor(pos / this.W);
            creatures[i].x = x; creatures[i].y = y;
            this.grid[pos] = creatures[i];
        }
        this.creatures = creatures;
    }

    move(creature, dx, dy) {
        const nx = Math.max(0, Math.min(this.W - 1, creature.x + dx));
        const ny = Math.max(0, Math.min(this.H - 1, creature.y + dy));
        if (nx === creature.x && ny === creature.y) return false;
        if (this.get(nx, ny) !== null) return false;
        this.set(creature.x, creature.y, null);
        creature.x = nx; creature.y = ny;
        this.set(nx, ny, creature);
        return true;
    }

    killAt(x, y) {
        const v = this.get(x, y);
        if (v && v.alive) { v.alive = false; this.set(x, y, null); this.murders++; }
    }

    localDensity(cx, cy, r = 2) {
        let count = 0;
        for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
            if (dx === 0 && dy === 0) continue;
            if (this.inBounds(cx + dx, cy + dy) && this.get(cx + dx, cy + dy)) count++;
        }
        return count;
    }

    stepCreature(creature, simStep, stepsPerGen, killEnabled) {
        if (!creature.alive) return;
        creature.age++;
        creature.oscPhase += (2 * Math.PI) / 30;

        // Sense
        const s = new Float32Array(NUM_SENSORS);
        // Recenter spatial inputs to [-1, 1] to remove East/South bias
        s[0] = (creature.x / (this.W - 1)) * 2 - 1;
        s[1] = (creature.y / (this.H - 1)) * 2 - 1;
        s[2] = creature.age / Math.max(1, stepsPerGen - 1);
        s[3] = this.rng() * 2 - 1; // Recenter random too
        s[4] = (Math.sin(creature.oscPhase) + 1) / 2;
        s[5] = 1 - Math.min(creature.x, this.W - 1 - creature.x) / (this.W / 2);
        s[6] = 1 - Math.min(creature.y, this.H - 1 - creature.y) / (this.H / 2);
        s[7] = Math.min(1, this.localDensity(creature.x, creature.y) / 8);
        const [ddx, ddy] = DIRS[creature.lastDir];
        let fwd = 0, bwd = 0;
        for (let step = 1; step <= 5; step++) {
            if (this.inBounds(creature.x + ddx * step, creature.y + ddy * step) && this.get(creature.x + ddx * step, creature.y + ddy * step)) fwd++;
            if (this.inBounds(creature.x - ddx * step, creature.y - ddy * step) && this.get(creature.x - ddx * step, creature.y - ddy * step)) bwd++;
        }
        s[8] = (fwd - bwd) / 5;
        const ahead = this.get(creature.x + ddx, creature.y + ddy);
        s[9] = ahead ? genomeSimilarity(creature.genome, ahead.genome) : 0;
        // Recenter move inputs to [-1, 1]
        s[10] = DIRS[creature.lastDir][0];
        s[11] = DIRS[creature.lastDir][1];
        s[12] = ahead ? 1 : 0;
        s[13] = 1;

        // Think
        const actions = creature.brain.forward(s);
        let bestAct = -1, bestVal = 0.1;
        for (let i = 0; i < NUM_ACTIONS; i++) {
            if (Math.abs(actions[i]) > bestVal) { bestVal = Math.abs(actions[i]); bestAct = i; }
        }
        if (bestAct < 0) return;

        // Act
        const str = actions[bestAct];
        if (bestAct === 0) { const dx2 = str > 0 ? 1 : -1; if (this.move(creature, dx2, 0)) { creature.lastDir = str > 0 ? 0 : 4; } }
        else if (bestAct === 1) { const dy2 = str > 0 ? 1 : -1; if (this.move(creature, 0, dy2)) { creature.lastDir = str > 0 ? 2 : 6; } }
        else if (bestAct === 2) { const d = Math.floor(this.rng() * 8); if (this.move(creature, DIRS[d][0], DIRS[d][1])) creature.lastDir = d; }
        else if (bestAct === 3) { this.move(creature, DIRS[creature.lastDir][0], DIRS[creature.lastDir][1]); }
        else if (bestAct === 4) { creature.lastDir = (creature.lastDir + 7) % 8; this.move(creature, DIRS[creature.lastDir][0], DIRS[creature.lastDir][1]); }
        else if (bestAct === 5) { creature.lastDir = (creature.lastDir + 1) % 8; this.move(creature, DIRS[creature.lastDir][0], DIRS[creature.lastDir][1]); }
        else if (bestAct === 6) { creature.lastDir = (creature.lastDir + 4) % 8; this.move(creature, DIRS[creature.lastDir][0], DIRS[creature.lastDir][1]); }
        else if (bestAct === 7 && killEnabled) { this.killAt(creature.x + DIRS[creature.lastDir][0], creature.y + DIRS[creature.lastDir][1]); }
    }
}
