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
  const csv  = rows.map(r => r.join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href     = url
  a.download = `ppe_violations_${Date.now()}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

export const ViolationLog: React.FC<Props> = ({ log, onClear }) => (
  <div className="bg-gray-900 rounded-xl p-5">
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center gap-2">
        <AlertTriangle className="w-4 h-4 text-red-400" />
        <h3 className="text-sm font-semibold text-gray-200">Violation Log</h3>
        {log.length > 0 && (
          <span className="bg-red-600 text-white text-xs rounded-full px-2 py-0.5">
            {log.length}
          </span>
        )}
      </div>
      {log.length > 0 && (
        <div className="flex gap-2">
          <button
            onClick={() => downloadCSV(log)}
            className="text-xs flex items-center gap-1 text-blue-400 hover:text-blue-300 transition-colors"
          >
            <Download className="w-3 h-3" /> CSV
          </button>
          <button
            onClick={onClear}
            className="text-xs flex items-center gap-1 text-gray-500 hover:text-red-400 transition-colors"
          >
            <Trash2 className="w-3 h-3" /> Clear
          </button>
        </div>
      )}
    </div>

    {log.length === 0 ? (
      <p className="text-sm text-gray-600 text-center py-6">No violations recorded yet.</p>
    ) : (
      <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
        {[...log].reverse().map(entry => (
          <div
            key={entry.id}
            className="flex gap-3 p-3 bg-red-950/40 border border-red-900/40 rounded-lg"
          >
            {entry.snapshot && (
              <img
                src={entry.snapshot}
                alt="snapshot"
                className="w-14 h-14 object-cover rounded flex-shrink-0"
              />
            )}
            <div className="min-w-0">
              <p className="text-xs text-gray-400">{entry.timestamp}</p>
              <p className="text-sm text-red-300 font-medium mt-0.5">
                Missing: {entry.missing.map(m => m.replace('no_', '')).join(', ')}
              </p>
            </div>
          </div>
        ))}
      </div>
    )}
  </div>
)
