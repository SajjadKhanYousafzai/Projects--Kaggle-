import React from 'react'
import { HardHat, Activity } from 'lucide-react'

interface Props {
  modelReady: boolean
  modelStatus: string
}

export const Header: React.FC<Props> = ({ modelReady, modelStatus }) => (
  <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
    <div className="max-w-7xl mx-auto flex items-center justify-between">
      <div className="flex items-center gap-3">
        <HardHat className="w-8 h-8 text-yellow-400" />
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight">
            PPE Kit Detection
          </h1>
          <p className="text-xs text-gray-400">
            Construction Site Safety · YOLO11s · 11 Classes
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2 text-sm">
        <Activity
          className={`w-4 h-4 ${modelReady ? 'text-green-400' : 'text-yellow-400 animate-pulse'}`}
        />
        <span className={modelReady ? 'text-green-400' : 'text-yellow-400'}>
          {modelStatus}
        </span>
      </div>
    </div>
  </header>
)
