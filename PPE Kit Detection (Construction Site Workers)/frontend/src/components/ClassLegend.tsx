import React from 'react'
import { CLASS_NAMES, CLASS_COLORS, POSITIVE_CLASSES, NEGATIVE_CLASSES } from '../types'

export const ClassLegend: React.FC = () => (
  <div className="glass-panel p-5 rounded-2xl">
    <p className="text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
      <span className="w-1.5 h-1.5 rounded-full bg-slate-500"></span>
      Detection System Classes
    </p>
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
      {CLASS_NAMES.map(name => {
        let badge = '⬜'
        let badgeClass = 'border-slate-500/30'
        if (POSITIVE_CLASSES.has(name)) { badge = '🟢'; badgeClass = 'border-emerald-500/30 bg-emerald-500/5' }
        if (NEGATIVE_CLASSES.has(name)) { badge = '🔴'; badgeClass = 'border-rose-500/30 bg-rose-500/5' }
        return (
          <div
            key={name}
            className={`flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs border bg-midnight-900/50 shadow-sm hover:bg-white/5 transition-colors ${badgeClass}`}
          >
            <span
              className="w-3.5 h-3.5 rounded-md flex-shrink-0 shadow-inner"
              style={{ backgroundColor: CLASS_COLORS[name] }}
            />
            <span className="text-slate-300 font-medium truncate flex-1">{name}</span>
            <span className="text-[10px] opacity-80">{badge}</span>
          </div>
        )
      })}
    </div>
  </div>
)
