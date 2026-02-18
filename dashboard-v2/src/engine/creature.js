/* ============================================================
   CREATURE MODULE
   ============================================================ */

import { NeuralNet } from './neuralNet';
import { genomeToColor } from './genome';


export class Creature {
    constructor(x, y, genome, maxInternal, rng) {
        this.x = x; this.y = y;
        this.genome = genome;
        this.brain = new NeuralNet(genome, maxInternal);
        this.alive = true;
        this.age = 0;
        // Randomize initial direction to avoid east-bias from move_forward
        this.lastDir = rng ? Math.floor(rng() * 8) : Math.floor(Math.random() * 8);
        this.oscPhase = 0;
        this.radiationDose = 0;
        this.color = genomeToColor(genome);
    }
}
