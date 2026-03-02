import React from 'react'
import { ViolationEntry } from '../types'
import { AlertTriangle, Trash2, Download } from 'lucide-react'

interface Props {
  log: ViolationEntry[]
  onClear: () => void
}

function downloadCSV(log: ViolationEntry[]) {
  const rows = [['id', 'timestamp', 'missing_ppe']]
  log.forEach(e => rows.push([String(e.id), e.timestamp, e.missing.join(' | ')]))
  const csv = rows.map(r => r.join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `ppe_violations_${Date.now()}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

export const ViolationLog: React.FC<Props> = ({ log, onClear }) => (
  <div className="glass-panel p-6 rounded-2xl flex flex-col h-[calc(100%-400px)] min-h-[300px]">
    <div className="flex items-center justify-between mb-6 pb-4 border-b border-white/5">
      <div className="flex items-center gap-3">
        <div className="p-1.5 bg-rose-500/10 rounded-lg border border-rose-500/20">
          <AlertTriangle className="w-4 h-4 text-rose-400" />
        </div>
        <h3 className="text-sm font-bold text-slate-200 tracking-wide uppercase">Incident Log</h3>
        {log.length > 0 && (
          <span className="bg-rose-500 text-white text-[10px] font-bold px-2 py-0.5 rounded-full shadow-sm">
            {log.length}
          </span>
        )}
      </div>
      {log.length > 0 && (
        <div className="flex gap-1.5">
          <button
            onClick={() => downloadCSV(log)}
            className="p-1.5 rounded-lg text-emerald-400 hover:bg-emerald-500/10 hover:text-emerald-300 transition-colors border border-transparent hover:border-emerald-500/20"
            title="Download CSV"
          >
            <Download className="w-4 h-4" />
          </button>
          <button
            onClick={onClear}
            className="p-1.5 rounded-lg text-slate-400 hover:bg-rose-500/10 hover:text-rose-400 transition-colors border border-transparent hover:border-rose-500/20"
            title="Clear Log"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>

    {log.length === 0 ? (
      <div className="flex-1 flex flex-col items-center justify-center text-center opacity-60">
        <AlertTriangle className="w-10 h-10 text-slate-600 mb-3" />
        <p className="text-sm text-slate-400 font-medium">No safety incidents recorded.</p>
        <p className="text-xs text-slate-500 mt-1">Violations will appear here in real-time.</p>
      </div>
    ) : (
      <div className="space-y-3 overflow-y-auto pr-2 custom-scrollbar flex-1">
        {[...log].reverse().map(entry => (
          <div
            key={entry.id}
            className="flex gap-4 p-3.5 bg-rose-500/5 hover:bg-rose-500/10 border border-rose-500/20 hover:border-rose-500/40 rounded-xl transition-colors animate-slide-up group"
          >
            {entry.snapshot && (
              <div className="relative w-16 h-16 rounded-lg overflow-hidden flex-shrink-0 border border-rose-500/30 shadow-sm">
                <img
                  src={entry.snapshot}
                  alt="Incident Snapshot"
                  className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                />
                <div className="absolute inset-0 ring-1 ring-inset ring-black/20 rounded-lg"></div>
              </div>
            )}
            <div className="min-w-0 flex-1 flex flex-col justify-center">
              <div className="flex items-center gap-2 mb-1">
                <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse"></span>
                <p className="text-xs font-mono text-slate-400">{entry.timestamp}</p>
              </div>
              <p className="text-sm text-rose-300 font-semibold leading-tight line-clamp-2">
                Missing: <span className="text-slate-300 font-medium">{entry.missing.map(m => m.replace('no_', '')).join(', ')}</span>
              </p>
            </div>
          </div>
        ))}
      </div>
    )}
  </div>
)
