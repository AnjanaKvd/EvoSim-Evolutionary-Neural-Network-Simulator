import React, { useRef, useEffect } from 'react';

// Pre-defined color palettes for different species/genomes could be cool
// but for now we stick to the RGB calculated from genome.

export function WorldRenderer({ world, generation, config }) {
    const canvasRef = useRef(null);
    const containerRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        const container = containerRef.current;
        if (!canvas || !container) return;

        const ro = new ResizeObserver(entries => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                canvas.width = width;
                canvas.height = height;
            }
        });
        ro.observe(container);
        return () => ro.disconnect();
    }, []);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !world) return;
        const ctx = canvas.getContext('2d');
        const { width, height } = canvas;
        const { W, H } = world;

        // Clear & Background
        ctx.fillStyle = '#050b14';
        ctx.fillRect(0, 0, width, height);

        if (width === 0 || height === 0) return;

        const cellW = width / W;
        const cellH = height / H;

        // Draw Grid (Subtle)
        ctx.strokeStyle = 'rgba(0, 240, 255, 0.05)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let x = 0; x <= W; x += 4) { ctx.moveTo(x * cellW, 0); ctx.lineTo(x * cellW, height); }
        for (let y = 0; y <= H; y += 4) { ctx.moveTo(0, y * cellH); ctx.lineTo(width, y * cellH); }
        ctx.stroke();

        // Draw Selection Zones (Glow effects)
        ctx.save();
        ctx.globalCompositeOperation = 'screen';
        const mode = config.selectionMode;

        if (mode === 'east') {
            ctx.fillStyle = 'rgba(0, 255, 153, 0.1)';
            ctx.fillRect(width / 2, 0, width / 2, height);
            ctx.strokeStyle = '#00ff99';
            ctx.strokeRect(width / 2, 0, width / 2, height);
        } else if (mode === 'west') {
            ctx.fillStyle = 'rgba(0, 255, 153, 0.1)';
            ctx.fillRect(0, 0, width / 2, height);
            ctx.strokeStyle = '#00ff99';
            ctx.strokeRect(0, 0, width / 2, height);
        } else if (mode === 'center') {
            ctx.fillStyle = 'rgba(0, 255, 153, 0.1)';
            ctx.beginPath();
            ctx.ellipse(width / 2, height / 2, config.centerRadius * cellW, config.centerRadius * cellH, 0, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#00ff99';
            ctx.stroke();
        } else if (mode === 'radioactive') {
            // Radioactive danger zones (Red Glow)
            const halfSteps = (config.stepsPerGen || 200) / 2;
            const lethalDist = Math.max(1, Math.ceil(-Math.log(1.0 / (0.03 * halfSteps)) / 0.04));

            const gradL = ctx.createLinearGradient(0, 0, lethalDist * cellW, 0);
            gradL.addColorStop(0, 'rgba(255, 0, 85, 0.4)');
            gradL.addColorStop(1, 'rgba(255, 0, 85, 0)');
            ctx.fillStyle = gradL;
            ctx.fillRect(0, 0, lethalDist * cellW, height);

            const gradR = ctx.createLinearGradient(width, 0, width - lethalDist * cellW, 0);
            gradR.addColorStop(0, 'rgba(255, 0, 85, 0.4)');
            gradR.addColorStop(1, 'rgba(255, 0, 85, 0)');
            ctx.fillStyle = gradR;
            ctx.fillRect(width - lethalDist * cellW, 0, lethalDist * cellW, height);
        }
        ctx.restore();

        // Draw Creatures (Sprites/Glows)
        world.creatures.forEach(c => {
            if (!c.alive) return;
            const cx = c.x * cellW + cellW / 2;
            const cy = c.y * cellH + cellH / 2;
            const radius = Math.min(cellW, cellH) * 0.4;

            ctx.save();
            ctx.translate(cx, cy);

            // Color
            const [r, g, b] = c.color;
            const colorStr = `rgb(${r},${g},${b})`;

            // Glow
            ctx.shadowColor = colorStr;
            ctx.shadowBlur = 10;

            // Creature Body (Triangle for direction or Circle)
            // Let's use a simple geometric shape that looks techy
            ctx.fillStyle = colorStr;
            ctx.beginPath();
            ctx.arc(0, 0, radius, 0, Math.PI * 2);
            ctx.fill();

            // Direction Indicator
            // const dirIdx = c.lastDir;
            // const angle = Math.atan2(DIRS[dirIdx][1], DIRS[dirIdx][0]); // Need DIRS imported or approximated
            // Actually, let's just draw a small white dot offset
            // We know DIRS from logic but haven't passed it. 
            // Let's just draw a center core
            ctx.fillStyle = '#fff';
            ctx.beginPath();
            ctx.arc(0, 0, radius * 0.4, 0, Math.PI * 2);
            ctx.fill();

            ctx.restore();
        });

    }, [world, generation, config]);

    return (
        <div ref={containerRef} className="w-full h-full relative rounded-lg overflow-hidden border border-[var(--border)] shadow-[0_0_20px_rgba(0,0,0,0.5)] bg-black">
            <canvas ref={canvasRef} className="block w-full h-full" />
            {/* Overlay Scanlines specific to canvas if needed, or global */}
            <div className="absolute top-2 left-2 text-[10px] uppercase font-mono text-[var(--primary)] opacity-70 pointer-events-none">
                Sector View: LIVE
            </div>
        </div>
    );
}
