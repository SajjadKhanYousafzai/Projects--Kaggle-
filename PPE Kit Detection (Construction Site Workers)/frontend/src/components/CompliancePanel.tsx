import React from 'react'
import { ComplianceResult, CLASS_COLORS, classType } from '../types'
import { CheckCircle, XCircle, ShieldCheck, ShieldX, Activity } from 'lucide-react'

interface Props {
  result: ComplianceResult | null
  loading?: boolean
}

export const CompliancePanel: React.FC<Props> = ({ result, loading }) => {
  if (loading) {
    return (
      <div className="glass-panel p-6 rounded-2xl flex flex-col items-center justify-center h-48 gap-4 border-emerald-500/20">
        <div className="w-10 h-10 rounded-full border-4 border-emerald-500/20 border-t-emerald-500 animate-spin"></div>
        <span className="text-emerald-400 font-medium animate-pulse">Analyzing frame...</span>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="glass-panel p-6 rounded-2xl flex flex-col items-center justify-center h-48 text-center gap-3">
        <Activity className="w-8 h-8 text-slate-600" />
        <p className="text-slate-500 text-sm max-w-[200px]">
          System standby. Awaiting visual input for analysis.
        </p>
      </div>
    )
  }

  return (
    <div className="glass-panel p-6 rounded-2xl space-y-6">
      {/* Overall status */}
      <div
        className={`flex items-center gap-4 p-4 rounded-xl shadow-lg border backdrop-blur-md transition-all duration-300 ${result.compliant
            ? 'bg-emerald-500/10 border-emerald-500/30'
            : 'bg-rose-500/10 border-rose-500/30 animate-pulse-slow'
          }`}
      >
        <div className={`p-2 rounded-lg ${result.compliant ? 'bg-emerald-500/20' : 'bg-rose-500/20'}`}>
          {result.compliant ? (
            <ShieldCheck className="w-6 h-6 text-emerald-400" />
          ) : (
            <ShieldX className="w-6 h-6 text-rose-400" />
          )}
        </div>
        <div>
          <p className={`font-bold text-sm tracking-wide ${result.compliant ? 'text-emerald-300' : 'text-rose-300'}`}>
            {result.compliant ? 'COMPLIANT STATUS' : 'VIOLATION DETECTED'}
          </p>
          <p className="text-xs text-slate-400 mt-1">
            {result.compliant ? 'All critical safety gear verified.' : 'Immediate action required.'}
          </p>
          {result.fps !== undefined && (
            <div className="flex items-center gap-2 mt-2">
              <span className="text-[10px] uppercase font-bold text-slate-500 bg-midnight-900 border border-white/5 px-2 py-0.5 rounded">
                {result.fps} FPS
              </span>
              <span className="text-[10px] uppercase font-bold text-slate-500 bg-midnight-900 border border-white/5 px-2 py-0.5 rounded">
                {result.detections.length} Obj
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Wearing (positive) */}
      {result.wearing.length > 0 && (
        <div>
          <p className="text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
            Verified Gear
          </p>
          <div className="flex flex-wrap gap-2">
            {result.wearing.map(item => (
              <span
                key={item}
                className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg font-medium shadow-sm"
                style={{
                  backgroundColor: CLASS_COLORS[item] + '20',
                  color: CLASS_COLORS[item],
                  border: `1px solid ${CLASS_COLORS[item]}40`
                }}
              >
                <CheckCircle className="w-3.5 h-3.5" />
                {item}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Missing (negative) */}
      {result.missing.length > 0 && (
        <div>
          <p className="text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-rose-500"></span>
            Missing Gear
          </p>
          <div className="flex flex-wrap gap-2">
            {result.missing.map(item => (
              <span
                key={item}
                className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg font-medium shadow-sm bg-rose-500/10 text-rose-400 border border-rose-500/30"
              >
                <XCircle className="w-3.5 h-3.5" />
                {item.replace('no_', 'Missing ')}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Per-class confidence bars */}
      {result.detections.length > 0 && (
        <div className="pt-2 border-t border-white/5">
          <p className="text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-4">
            Detection Confidence
          </p>
          <div className="space-y-3 max-h-48 overflow-y-auto pr-2 custom-scrollbar">
            {result.detections
              .sort((a, b) => b.confidence - a.confidence)
              .map((d, i) => (
                <div key={i} className="flex flex-col gap-1">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-slate-300">{d.className}</span>
                    <span className="text-xs font-mono text-slate-400">
                      {(d.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1.5 w-full bg-midnight-900 rounded-full overflow-hidden border border-white/5">
                    <div
                      className="h-full rounded-full transition-all duration-300 relative"
                      style={{
                        width: `${d.confidence * 100}%`,
                        backgroundColor: CLASS_COLORS[d.className] ?? '#64748b',
                      }}
                    >
                      <div className="absolute inset-0 bg-gradient-to-r from-transparent to-white/20"></div>
                    </div>
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  )
}
