import React from 'react';
import { Settings, Sliders, Zap, Crosshair, Users, Activity } from 'lucide-react';

export function SidebarControls({ config, setConfig }) {
    const update = (key, val) => setConfig(prev => ({ ...prev, [key]: val }));

    return (
        <div className="space-y-6 font-mono text-sm">

            {/* World Section */}
            <Section title="World Parameters" icon={Sliders}>
                <Control label="Width" value={config.worldWidth} min={32} max={128} step={8} onChange={v => update('worldWidth', v)} />
                <Control label="Height" value={config.worldHeight} min={32} max={128} step={8} onChange={v => update('worldHeight', v)} />
                <Control label="Population" value={config.population} min={50} max={2000} step={50} onChange={v => update('population', v)} />
            </Section>

            {/* Simulation Section */}
            <Section title="Simulation Config" icon={Zap}>
                <Control label="Steps/Gen" value={config.stepsPerGen} min={50} max={500} step={25} onChange={v => update('stepsPerGen', v)} />
                <Control label="Mutation Rate" value={config.mutationRate} min={0} max={0.01} step={0.0001} smallStep onChange={v => update('mutationRate', v)} />
                <Control label="Genome Size" value={config.genomeSize} min={2} max={64} step={2} onChange={v => update('genomeSize', v)} />
                <Control label="Internal Neurons" value={config.maxInternal} min={1} max={8} step={1} onChange={v => update('maxInternal', v)} />
            </Section>

            {/* Selection Section */}
            <Section title="Evolution Pressure" icon={Crosshair}>
                <div className="mb-4">
                    <label className="block text-[10px] uppercase tracking-wider text-[var(--accent)] mb-1">Target Scenario</label>
                    <select
                        value={config.selectionMode}
                        onChange={e => update('selectionMode', e.target.value)}
                        className="w-full bg-[var(--bg-dark)] border border-[var(--border)] text-[var(--primary)] rounded text-xs p-2 focus:ring-1 focus:ring-[var(--primary)] outline-none cursor-pointer"
                    >
                        <option value="east">East Migration</option>
                        <option value="west">West Migration</option>
                        <option value="west_east">Divergent (East/West)</option>
                        <option value="corners">Corner Seeking</option>
                        <option value="center">Center Convergence</option>
                        <option value="radioactive">Radioactive Evasion</option>
                    </select>
                    <div className="mt-2 text-[10px] text-[var(--text-muted)] italic leading-tight">
                        {getScenarioDesc(config.selectionMode)}
                    </div>
                </div>
                <Toggle label="Kill Mechanism" active={config.killEnabled} onClick={() => update('killEnabled', !config.killEnabled)} />
            </Section>

        </div>
    );
}

function Section({ title, icon: Icon, children }) {
    return (
        <div className="space-y-3">
            <div className="flex items-center gap-2 text-[var(--secondary)] border-b border-[var(--border)] pb-1 mb-2">
                <Icon size={14} />
                <h3 className="text-xs font-bold uppercase tracking-widest">{title}</h3>
            </div>
            {children}
        </div>
    );
}

function Control({ label, value, min, max, step, onChange, smallStep }) {
    return (
        <div className="group">
            <div className="flex justify-between mb-1">
                <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider group-hover:text-[var(--text-main)] transition-colors">{label}</span>
                <span className="text-[10px] font-bold text-[var(--primary)] font-mono">{smallStep ? value.toFixed(4) : value}</span>
            </div>
            <input
                type="range" min={min} max={max} step={step} value={value}
                onChange={e => onChange(parseFloat(e.target.value))}
                className="w-full h-1 bg-[var(--bg-dark)] rounded-lg appearance-none cursor-pointer range-slider"
            />
        </div>
    );
}

function Toggle({ label, active, onClick }) {
    return (
        <div className="flex justify-between items-center cursor-pointer" onClick={onClick}>
            <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">{label}</span>
            <div className={`w-8 h-4 rounded-full relative transition-colors ${active ? 'bg-[var(--danger)]' : 'bg-[var(--bg-dark)]'}`}>
                <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform ${active ? 'left-4.5' : 'left-0.5'}`} />
            </div>
        </div>
    )
}

function getScenarioDesc(mode) {
    const map = {
        east: "Survive in right half → Evolves eastward migration.",
        west: "Survive in left half → Evolves westward migration.",
        west_east: "Survive in side strips → Species Split.",
        corners: "Survive in corners → Corner-seeking behavior.",
        center: "Survive in center → Convergent behavior.",
        radioactive: "Avoid radioactive walls → Midline crossing.",
    };
    return map[mode] || "";
}
