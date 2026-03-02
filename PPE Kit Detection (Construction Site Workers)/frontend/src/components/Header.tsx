import React from 'react'
import { HardHat, Activity } from 'lucide-react'

interface Props {
  modelReady: boolean
  modelStatus: string
}

export const Header: React.FC<Props> = ({ modelReady, modelStatus }) => (
  <header className="glass-header px-6 py-4 sticky top-0 z-50">
    <div className="max-w-7xl mx-auto flex items-center justify-between">
      <div className="flex items-center gap-4">
        <div className="relative">
          <div className="absolute inset-0 bg-emerald-500/20 blur-xl rounded-full"></div>
          <div className="bg-midnight-800 p-2.5 rounded-xl border border-white/5 relative">
            <HardHat className="w-8 h-8 text-emerald-400 drop-shadow-[0_0_8px_rgba(52,211,153,0.5)]" />
          </div>
        </div>
        <div>
          <h1 className="text-2xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-teal-200">
            PPE Kit Detection
          </h1>
          <p className="text-xs text-slate-400 font-medium">
            AI-Powered Construction Site Safety
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2 text-sm bg-midnight-800/50 px-3 py-1.5 rounded-full border border-white/5">
        <Activity
          className={`w-4 h-4 ${modelReady ? 'text-emerald-400' : 'text-amber-400 animate-pulse'}`}
        />
        <span className={`${modelReady ? 'text-emerald-400' : 'text-amber-400'} font-medium`}>
          {modelStatus}
        </span>
      </div>
    </div>
  </header>
)
