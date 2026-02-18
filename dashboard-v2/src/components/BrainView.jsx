import React, { useMemo } from 'react';
import { Network } from 'lucide-react';

export function BrainView({ creature }) {
    if (!creature) {
        return (
            <div className="h-full flex flex-col items-center justify-center text-[var(--text-muted)] opacity-50">
                <Network className="w-12 h-12 mb-2" />
                <span className="text-xs uppercase tracking-widest">No Specimen Selected</span>
            </div>
        );
    }

    const { conns } = creature.brain;

    // Reuse logic to determine active nodes
    const { sensors, internal, actions, edges } = useMemo(() => {
        const activeSensors = [...new Set(conns.filter(c => c.sourceType === 0).map(c => c.sourceId))].sort((a, b) => a - b);
        const activeInternal = [...new Set([
            ...conns.filter(c => c.sourceType === 1).map(c => c.sourceId),
            ...conns.filter(c => c.sinkType === 0).map(c => c.sinkId)
        ])].sort((a, b) => a - b);
        const activeActions = [...new Set(conns.filter(c => c.sinkType === 1).map(c => c.sinkId))].sort((a, b) => a - b);

        const posMap = {};
        const yPos = (arr, idx) => arr.length <= 1 ? 0.5 : idx / (arr.length - 1);

        activeSensors.forEach((id, i) => { posMap[`S${id}`] = { x: 0.1, y: 0.1 + 0.8 * yPos(activeSensors, i) }; });
        activeInternal.forEach((id, i) => { posMap[`I${id}`] = { x: 0.5, y: 0.1 + 0.8 * yPos(activeInternal, i) }; });
        activeActions.forEach((id, i) => { posMap[`A${id}`] = { x: 0.9, y: 0.1 + 0.8 * yPos(activeActions, i) }; });

        return {
            sensors: activeSensors,
            internal: activeInternal,
            actions: activeActions,
            edges: conns.map(c => {
                const sk = c.sourceType === 0 ? `S${c.sourceId}` : `I${c.sourceId}`;
                const tk = c.sinkType === 1 ? `A${c.sinkId}` : `I${c.sinkId}`;
                const sp = posMap[sk];
                const tp = posMap[tk];
                if (!sp || !tp) return null;
                return {
                    x1: sp.x, y1: sp.y, x2: tp.x, y2: tp.y,
                    weight: c.weight,
                    id: `${sk}-${tk}`
                };
            }).filter(Boolean),
            posMap
        };
    }, [creature]);

    return (
        <div className="w-full h-64 relative bg-[var(--bg-dark)] border border-[var(--border)] rounded overflow-hidden">
            <svg className="w-full h-full">
                {/* Edges */}
                {edges.map((e, i) => (
                    <line
                        key={i}
                        x1={`${e.x1 * 100}%`} y1={`${e.y1 * 100}%`}
                        x2={`${e.x2 * 100}%`} y2={`${e.y2 * 100}%`}
                        stroke={e.weight > 0 ? 'var(--primary)' : 'var(--danger)'}
                        strokeWidth={Math.abs(e.weight) * 2}
                        opacity={0.4}
                    />
                ))}

                {/* Nodes */}
                {sensors.map(id => <Node key={`S${id}`} x={0.1} y={edges.find(e => e.id.startsWith(`S${id}`))?.y1 || 0} color="var(--primary)" />)}
                {/* Wait, positions need to be retrieved from posMap, but posMap is internal to useMemo. 
                Values in edges are relative. Let's just re-iterate or pass posMap. 
                Actually, let's simplify. I already calculated pos for edges. 
                I should export nodes with positions from useMemo.
            */}
            </svg>

            {/* Re-implementing with absolute dict for clarity */}
            <div className="absolute inset-0 pointer-events-none">
                {Object.entries(actions.reduce((acc, id, i, arr) => ({ ...acc, [`A${id}`]: 0.1 + 0.8 * (arr.length <= 1 ? 0.5 : i / (arr.length - 1)) }), {})).map(([key, y]) => (
                    <div key={key} style={{ left: '90%', top: `${y * 100}%` }} className="absolute w-2 h-2 -ml-1 -mt-1 rounded-full bg-[var(--accent)] shadow-[0_0_5px_var(--accent)]" />
                ))}
                {Object.entries(sensors.reduce((acc, id, i, arr) => ({ ...acc, [`S${id}`]: 0.1 + 0.8 * (arr.length <= 1 ? 0.5 : i / (arr.length - 1)) }), {})).map(([key, y]) => (
                    <div key={key} style={{ left: '10%', top: `${y * 100}%` }} className="absolute w-2 h-2 -ml-1 -mt-1 rounded-full bg-[var(--primary)] shadow-[0_0_5px_var(--primary)]" />
                ))}
                {Object.entries(internal.reduce((acc, id, i, arr) => ({ ...acc, [`I${id}`]: 0.1 + 0.8 * (arr.length <= 1 ? 0.5 : i / (arr.length - 1)) }), {})).map(([key, y]) => (
                    <div key={key} style={{ left: '50%', top: `${y * 100}%` }} className="absolute w-2 h-2 -ml-1 -mt-1 rounded-full bg-[var(--text-muted)]" />
                ))}
            </div>

            <div className="absolute bottom-1 right-2 text-[9px] text-[var(--text-muted)] uppercase tracking-wider">
                Active Synapses: {conns.length}
            </div>
        </div>
    );
}

function Node({ x, y, color }) {
    // This helper was unused above because I used absolute divs over SVG for nodes themselves to easily add effects
    // But keeping it just in case.
    return <circle cx={`${x * 100}%`} cy={`${y * 100}%`} r="3" fill={color} />
}
