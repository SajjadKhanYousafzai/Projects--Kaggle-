import React, { useRef, useEffect, useCallback, useState } from 'react'
import Webcam from 'react-webcam'
import { detect } from '../utils/detector'
import { drawDetections } from '../utils/drawing'
import { ComplianceResult, ViolationEntry } from '../types'
import { CompliancePanel } from './CompliancePanel'
import { ViolationLog } from './ViolationLog'
import { Camera, CameraOff, Bell } from 'lucide-react'

interface Props {
  modelReady: boolean
  confThresh: number
}

export const WebcamTab: React.FC<Props> = ({ modelReady, confThresh }) => {
  const webcamRef  = useRef<Webcam>(null)
  const canvasRef  = useRef<HTMLCanvasElement>(null)
  const rafRef     = useRef<number>(0)
  const runningRef = useRef(false)

  const [isRunning,   setIsRunning]   = useState(false)
  const [result,      setResult]      = useState<ComplianceResult | null>(null)
  const [violations,  setViolations]  = useState<ViolationEntry[]>([])
  const [alertOn,     setAlertOn]     = useState(true)
  const lastViolRef   = useRef(0)

  const loop = useCallback(async () => {
    if (!runningRef.current) return
    const video = webcamRef.current?.video
    const canvas = canvasRef.current
    if (video && canvas && video.readyState === 4) {
      try {
        const res = await detect(video, confThresh)
        drawDetections(canvas, video, res.detections)
        setResult(res)

        // Log violation (debounce 3 s)
        if (!res.compliant && Date.now() - lastViolRef.current > 3000) {
          lastViolRef.current = Date.now()
          // Snapshot
          const snap = document.createElement('canvas')
          snap.width  = 120
          snap.height = 80
          snap.getContext('2d')!.drawImage(canvas, 0, 0, 120, 80)
          const snapshot = snap.toDataURL('image/jpeg', 0.6)

          setViolations(prev => [
            ...prev,
            {
              id:        prev.length + 1,
              timestamp: new Date().toLocaleTimeString(),
              missing:   res.missing,
              snapshot,
            },
          ])
        }
      } catch (_) { /* ignore single-frame errors */ }
    }
    rafRef.current = requestAnimationFrame(loop)
  }, [confThresh])

  const start = () => {
    runningRef.current = true
    setIsRunning(true)
    rafRef.current = requestAnimationFrame(loop)
  }

  const stop = () => {
    runningRef.current = false
    cancelAnimationFrame(rafRef.current)
    setIsRunning(false)
  }

  useEffect(() => () => { runningRef.current = false; cancelAnimationFrame(rafRef.current) }, [])

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Camera feed */}
      <div className="lg:col-span-2 space-y-4">
        <div className="relative bg-black rounded-xl overflow-hidden aspect-video">
          <Webcam
            ref={webcamRef}
            className="absolute inset-0 w-full h-full object-cover opacity-0"
            videoConstraints={{ facingMode: 'user', width: 1280, height: 720 }}
            mirrored
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full object-contain"
          />
          {!isRunning && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-950/80">
              <p className="text-gray-400 text-sm">
                {modelReady ? 'Press Start to begin detection' : 'Loading model…'}
              </p>
            </div>
          )}
          {/* Live badge */}
          {isRunning && (
            <div className="absolute top-3 left-3 flex items-center gap-1.5 bg-red-600 text-white text-xs font-bold px-2 py-1 rounded-full">
              <span className="w-1.5 h-1.5 bg-white rounded-full animate-ping" />
              LIVE
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-3">
          <button
            disabled={!modelReady || isRunning}
            onClick={start}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-500 disabled:opacity-40 disabled:cursor-not-allowed rounded-lg text-sm font-medium transition-colors"
          >
            <Camera className="w-4 h-4" /> Start Detection
          </button>
          <button
            disabled={!isRunning}
            onClick={stop}
            className="flex items-center gap-2 px-4 py-2 bg-red-700 hover:bg-red-600 disabled:opacity-40 disabled:cursor-not-allowed rounded-lg text-sm font-medium transition-colors"
          >
            <CameraOff className="w-4 h-4" /> Stop
          </button>
          <button
            onClick={() => setAlertOn(v => !v)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${alertOn ? 'bg-yellow-600 hover:bg-yellow-500' : 'bg-gray-700 hover:bg-gray-600'}`}
          >
            <Bell className="w-4 h-4" /> {alertOn ? 'Alert On' : 'Alert Off'}
          </button>
        </div>
      </div>

      {/* Right panel */}
      <div className="space-y-4">
        <CompliancePanel result={result} />
        <ViolationLog log={violations} onClear={() => setViolations([])} />
      </div>
    </div>
  )
}
