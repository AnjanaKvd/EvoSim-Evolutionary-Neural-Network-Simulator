import React from 'react';
import { LineChart, Line, YAxis, ResponsiveContainer } from 'recharts';

export function StatsPanel({ stats, history }) {
    return (
        <div className="space-y-6">

            {/* Live Metrics Grid */}
            <div className="grid grid-cols-2 gap-2">
                <StatCard label="Survivors" value={stats.survivors} color="var(--success)" />
                <StatCard label="Population" value={stats.population} color="var(--text-main)" />
                <StatCard label="Diversity" value={stats.diversity} color="var(--secondary)" />
                <StatCard label="Survival Rate" value={`${stats.survivalPct}%`} color="var(--primary)" />
                <StatCard label="Murders" value={stats.murders} color="var(--danger)" />
            </div>

            {/* Population History */}
            <div className="w-full h-32 bg-[var(--bg-dark)] border border-[var(--border)] rounded">
                <div className="text-[8px] px-2 pt-1 text-[var(--text-muted)] uppercase tracking-widest">
                    Population History
                </div>

                {/* Wrapper with no padding on measured element */}
                <div className="w-full h-[calc(100%-1rem)] px-2">
                    <ResponsiveContainer width="100%" height={250}>
                        <LineChart data={history}>
                            <Line
                                type="monotone"
                                dataKey="survivors"
                                stroke="var(--success)"
                                strokeWidth={2}
                                dot={false}
                                isAnimationActive={false}
                            />
                            <YAxis hide domain={[0, 'auto']} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Diversity History */}
            <div className="w-full h-32 bg-[var(--bg-dark)] border border-[var(--border)] rounded">
                <div className="text-[8px] px-2 pt-1 text-[var(--text-muted)] uppercase tracking-widest">
                    Genetic Diversity
                </div>

                <div className="w-full h-[calc(100%-1rem)] px-2">
                    <ResponsiveContainer width="100%" height={250}>
                        <LineChart data={history}>
                            <Line
                                type="monotone"
                                dataKey="diversity"
                                stroke="var(--secondary)"
                                strokeWidth={2}
                                dot={false}
                                isAnimationActive={false}
                            />
                            <YAxis hide domain={[0, 100]} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

        </div>
    );
}

function StatCard({ label, value, color }) {
    return (
        <div className="bg-[var(--bg-dark)] border border-[var(--border)] p-2 rounded flex flex-col items-center justify-center neo-stat-card">
            <span className="text-[9px] uppercase tracking-widest text-[var(--text-muted)] mb-1">
                {label}
            </span>
            <span className="text-xl font-bold font-mono text-glow" style={{ color }}>
                {value}
            </span>
        </div>
    );
}
