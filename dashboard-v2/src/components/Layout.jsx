import React from 'react';
import { Activity, Cpu, Settings, Play, Pause, RotateCcw, FastForward } from 'lucide-react';

export function Layout({ children, sidebar, stats, header }) {
    return (
        <div className="h-screen w-screen flex flex-col bg-[var(--bg-dark)] text-[var(--text-main)] scanlines relative overflow-hidden">
            {/* Top Header / HUD */}
            <header className="h-14 border-b border-[var(--border)] flex items-center justify-between px-4 bg-[var(--bg-panel)] z-10 shrink-0">
                <div className="flex items-center gap-2">
                    <Cpu className="w-6 h-6 text-[var(--primary)]" />
                    <h1 className="text-xl font-bold tracking-widest text-[var(--primary)] text-glow font-[var(--font-mono)]">
                        NEURO<span className="text-white">GENESIS</span>
                    </h1>
                </div>
                <div className="flex items-center gap-4">
                    {header}
                </div>
            </header>

            {/* Main Content Area */}
            <div className="flex flex-1 overflow-hidden relative">
                {/* Left Sidebar - Controls */}
                <aside className="w-64 border-r border-[var(--border)] bg-[var(--bg-panel)]/50 backdrop-blur flex flex-col z-10 shrink-0">
                    <div className="p-3 border-b border-[var(--border)] uppercase text-xs font-bold text-[var(--text-muted)] tracking-wider flex items-center gap-2">
                        <Settings className="w-4 h-4" /> System Config
                    </div>
                    <div className="flex-1 overflow-y-auto p-4 space-y-6">
                        {sidebar}
                    </div>
                </aside>

                {/* Center - Viewport */}
                <main className="flex-1 relative flex flex-col min-w-0 bg-black/20">
                    {children}
                </main>

                {/* Right Sidebar - Stats & Analysis */}
                <aside className="w-80 border-l border-[var(--border)] bg-[var(--bg-panel)]/50 backdrop-blur flex flex-col z-10 shrink-0">
                    <div className="p-3 border-b border-[var(--border)] uppercase text-xs font-bold text-[var(--text-muted)] tracking-wider flex items-center gap-2">
                        <Activity className="w-4 h-4" /> Telemetry
                    </div>
                    <div className="flex-1 overflow-y-auto p-4">
                        {stats}
                    </div>
                </aside>
            </div>

            {/* Bottom Status Bar */}
            <footer className="h-6 border-t border-[var(--border)] bg-[var(--bg-panel)] text-[10px] flex items-center px-4 justify-between text-[var(--text-muted)] font-[var(--font-mono)] z-10 shrink-0">
                <div>SYSTEM STATUS: ONLINE</div>
                <div>V2.0.0-ALPHA</div>
            </footer>
        </div>
    );
}

export function Button({ icon: Icon, label, active, onClick, danger }) {
    return (
        <button
            onClick={onClick}
            className={`
        flex items-center gap-2 px-3 py-1.5 rounded text-xs font-bold uppercase tracking-wider transition-all
        border border-[var(--border)]
        ${active ? 'bg-[var(--primary-dim)] border-[var(--primary)] text-[var(--primary)] shadow-[0_0_10px_var(--primary-dim)]' : 'hover:bg-[var(--bg-panel-hover)] text-[var(--text-muted)] hover:text-white'}
        ${danger ? 'hover:text-[var(--danger)] hover:border-[var(--danger)]' : ''}
      `}
        >
            {Icon && <Icon className="w-4 h-4" />}
            {label}
        </button>
    );
}
