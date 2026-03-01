import React from 'react'
import { ComplianceResult, CLASS_COLORS, classType } from '../types'
import { CheckCircle, XCircle, ShieldCheck, ShieldX } from 'lucide-react'

interface Props {
  result: ComplianceResult | null
  loading?: boolean
}

export const CompliancePanel: React.FC<Props> = ({ result, loading }) => {
  if (loading) {
    return (
      <div className="bg-gray-900 rounded-xl p-5 flex items-center justify-center h-48">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-yellow-400" />
      </div>
    )
  }

  if (!result) {
    return (
      <div className="bg-gray-900 rounded-xl p-5 flex items-center justify-center h-48 text-gray-500 text-sm">
        No detection yet — point a camera or upload an image.
      </div>
    )
  }

  return (
    <div className="bg-gray-900 rounded-xl p-5 space-y-4">
      {/* Overall status */}
      <div
        className={`flex items-center gap-3 p-3 rounded-lg ${
          result.compliant
            ? 'bg-green-950 border border-green-700'
            : 'bg-red-950 border border-red-700'
        }`}
      >
        {result.compliant ? (
          <ShieldCheck className="w-6 h-6 text-green-400 flex-shrink-0" />
        ) : (
          <ShieldX className="w-6 h-6 text-red-400 flex-shrink-0" />
        )}
        <div>
          <p className={`font-bold text-sm ${result.compliant ? 'text-green-300' : 'text-red-300'}`}>
            {result.compliant ? 'COMPLIANT — All critical PPE present' : 'NON-COMPLIANT — Immediate action required'}
          </p>
          {result.fps !== undefined && (
            <p className="text-xs text-gray-400 mt-0.5">{result.fps} FPS · {result.detections.length} detections</p>
          )}
        </div>
      </div>

      {/* Wearing (positive) */}
      {result.wearing.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            ✅ PPE Detected
          </p>
          <div className="flex flex-wrap gap-2">
            {result.wearing.map(item => (
              <span
                key={item}
                className="flex items-center gap-1 text-xs px-2 py-1 rounded-full font-medium"
                style={{ backgroundColor: CLASS_COLORS[item] + '33', color: CLASS_COLORS[item], border: `1px solid ${CLASS_COLORS[item]}55` }}
              >
                <CheckCircle className="w-3 h-3" />
                {item}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Missing (negative) */}
      {result.missing.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            ❌ PPE Missing
          </p>
          <div className="flex flex-wrap gap-2">
            {result.missing.map(item => (
              <span
                key={item}
                className="flex items-center gap-1 text-xs px-2 py-1 rounded-full font-medium"
                style={{ backgroundColor: '#dc262633', color: '#f87171', border: '1px solid #dc262655' }}
              >
                <XCircle className="w-3 h-3" />
                {item.replace('no_', 'Missing ')}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Per-class confidence bars */}
      {result.detections.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Detection Scores
          </p>
          <div className="space-y-1.5 max-h-40 overflow-y-auto pr-1">
            {result.detections
              .sort((a, b) => b.confidence - a.confidence)
              .map((d, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-xs w-24 truncate text-gray-300">{d.className}</span>
                  <div className="flex-1 bg-gray-800 rounded-full h-1.5">
                    <div
                      className="h-1.5 rounded-full transition-all"
                      style={{
                        width: `${d.confidence * 100}%`,
                        backgroundColor: CLASS_COLORS[d.className] ?? '#6b7280',
                      }}
                    />
                  </div>
                  <span className="text-xs text-gray-400 w-8 text-right">
                    {(d.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  )
}
