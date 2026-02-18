/* ============================================================
   GENOME MODULE (Updated)
   ============================================================ */

export function randomGenome(size, rng) {
    return Array.from({ length: size }, () => ((rng() * 0xFFFFFFFF) | 0));
}

export function decodeGene(gene, numSensors, numActions, maxInternal) {
    const u = gene >>> 0;
    const sourceType = (u >>> 31) & 1;
    let sourceId = (u >>> 24) & 0x7F;
    const sinkType = (u >>> 23) & 1;
    let sinkId = (u >>> 16) & 0x7F;
    let rawW = u & 0xFFFF;
    if (rawW >= 0x8000) rawW -= 0x10000;
    const weight = rawW / 8000;
    sourceId = sourceType === 0 ? sourceId % numSensors : sourceId % Math.max(1, maxInternal);
    sinkId = sinkType === 1 ? sinkId % numActions : sinkId % Math.max(1, maxInternal);
    return { sourceType, sourceId, sinkType, sinkId, weight };
}

export function mutateGenome(genome, rate, rng) {
    return genome.map(gene => {
        let g = gene >>> 0;
        for (let b = 0; b < 32; b++) if (rng() < rate) g ^= (1 << b);
        g = g >>> 0;
        return g >= 0x80000000 ? g - 0x100000000 : g;
    });
}

export function crossover(ga, gb, rng) {
    const split = Math.floor(rng() * (ga.length + 1));
    return [...ga.slice(0, split), ...gb.slice(split)];
}

export function popcount(n) {
    n = n >>> 0;
    n -= (n >> 1) & 0x55555555;
    n = (n & 0x33333333) + ((n >> 2) & 0x33333333);
    n = (n + (n >> 4)) & 0x0f0f0f0f;
    return (n * 0x01010101) >>> 24;
}

export function genomeSimilarity(ga, gb) {
    let match = 0, total = 0;
    for (let i = 0; i < Math.min(ga.length, gb.length); i++) {
        const xor = (ga[i] ^ gb[i]) >>> 0;
        match += 32 - popcount(xor); total += 32;
    }
    return total ? match / total : 1;
}

export function genomeToColor(genome) {
    let h = 0;
    for (const g of genome) h ^= (g >>> 0) & 0xFFFFFF;
    let r = (h >> 16) & 0xFF, g = (h >> 8) & 0xFF, b = h & 0xFF;
    return [Math.max(60, r), Math.max(60, g), Math.max(60, b)];
}

export function reproduce(survivors, pop, genomeSize, mutationRate, rng) {
    if (!survivors.length) return Array.from({ length: pop }, () => randomGenome(genomeSize, rng));
    const next = [];
    while (next.length < pop) {
        const pa = survivors[Math.floor(rng() * survivors.length)];
        const pb = survivors[Math.floor(rng() * survivors.length)];
        let child = crossover(pa.genome, pb.genome, rng);
        child = mutateGenome(child, mutationRate, rng);
        next.push(child);
    }
    return next;
}

export function calcDiversity(survivors) {
    if (survivors.length < 2) return 0;
    const sample = survivors.slice(0, Math.min(30, survivors.length));
    let total = 0, count = 0;
    for (let i = 0; i < sample.length; i++)
        for (let j = i + 1; j < sample.length; j++) {
            total += 1 - genomeSimilarity(sample[i].genome, sample[j].genome);
            count++;
        }
    return count ? total / count : 0;
}
