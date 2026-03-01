import React from 'react'
import { CLASS_NAMES, CLASS_COLORS, POSITIVE_CLASSES, NEGATIVE_CLASSES } from '../types'

export const ClassLegend: React.FC = () => (
  <div className="bg-gray-900 rounded-xl p-4">
    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
      Detection Classes
    </p>
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
      {CLASS_NAMES.map(name => {
        let badge = '⬜'
        if (POSITIVE_CLASSES.has(name)) badge = '🟢'
        if (NEGATIVE_CLASSES.has(name)) badge = '🔴'
        return (
          <div
            key={name}
            className="flex items-center gap-1.5 text-xs"
          >
            <span
              className="w-3 h-3 rounded-sm flex-shrink-0"
              style={{ backgroundColor: CLASS_COLORS[name] }}
            />
            <span className="text-gray-300 truncate">{name}</span>
            <span className="text-xs">{badge}</span>
          </div>
        )
      })}
    </div>
  </div>
)
