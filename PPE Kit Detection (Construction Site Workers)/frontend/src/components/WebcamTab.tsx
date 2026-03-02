import React, { useRef, useEffect, useCallback, useState } from 'react'
import Webcam from 'react-webcam'
import { detect } from '../utils/detector'
import { drawDetections } from '../utils/drawing'
import { ComplianceResult, ViolationEntry } from '../types'
import { CompliancePanel } from './CompliancePanel'
import { ViolationLog } from './ViolationLog'
import { Camera, CameraOff, Bell, BellOff } from 'lucide-react'

interface Props {
  modelReady: boolean
  confThresh: number
}

export const WebcamTab: React.FC<Props> = ({ modelReady, confThresh }) => {
  const webcamRef = useRef<Webcam>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number>(0)
  const runningRef = useRef(false)

  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState<ComplianceResult | null>(null)
  const [violations, setViolations] = useState<ViolationEntry[]>([])
  const [alertOn, setAlertOn] = useState(true)
  const lastViolRef = useRef(0)

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
          snap.width = 120
          snap.height = 80
          snap.getContext('2d')!.drawImage(canvas, 0, 0, 120, 80)
          const snapshot = snap.toDataURL('image/jpeg', 0.6)

          setViolations(prev => [
            ...prev,
            {
              id: prev.length + 1,
              timestamp: new Date().toLocaleTimeString(),
              missing: res.missing,
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
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      {/* Camera feed */}
      <div className="lg:col-span-2 space-y-6">
        <div className="relative bg-midnight-900 rounded-2xl overflow-hidden aspect-video shadow-2xl border border-white/10 group">
          <Webcam
            ref={webcamRef}
            className="absolute inset-0 w-full h-full object-cover opacity-0"
            videoConstraints={{ facingMode: 'user', width: 1280, height: 720 }}
            mirrored
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
          />

          {/* Decorative Corner Accents */}
          <div className="absolute top-0 left-0 w-16 h-16 border-t-2 border-l-2 border-emerald-500/50 rounded-tl-xl m-4 pointer-events-none transition-opacity opacity-0 group-hover:opacity-100"></div>
          <div className="absolute top-0 right-0 w-16 h-16 border-t-2 border-r-2 border-emerald-500/50 rounded-tr-xl m-4 pointer-events-none transition-opacity opacity-0 group-hover:opacity-100"></div>
          <div className="absolute bottom-0 left-0 w-16 h-16 border-b-2 border-l-2 border-emerald-500/50 rounded-bl-xl m-4 pointer-events-none transition-opacity opacity-0 group-hover:opacity-100"></div>
          <div className="absolute bottom-0 right-0 w-16 h-16 border-b-2 border-r-2 border-emerald-500/50 rounded-br-xl m-4 pointer-events-none transition-opacity opacity-0 group-hover:opacity-100"></div>

          {!isRunning && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-midnight-900/90 backdrop-blur-sm gap-4">
              <div className="w-16 h-16 rounded-full bg-midnight-800 flex items-center justify-center border border-white/5 relative shadow-inner">
                {modelReady ? (
                  <Camera className="w-8 h-8 text-emerald-400" />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-8 h-8 rounded-full border-2 border-amber-400/30 border-t-amber-400 animate-spin"></div>
                  </div>
                )}
              </div>
              <p className="text-slate-400 font-medium text-sm">
                {modelReady ? 'Press Start to ignite detection engine' : 'Initializing AI models...'}
              </p>
            </div>
          )}

          {/* Live badge */}
          {isRunning && (
            <div className="absolute top-4 left-4 flex items-center gap-2 bg-rose-500/90 backdrop-blur-md text-white px-3 py-1.5 rounded-full shadow-lg border border-rose-400/30">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-white"></span>
              </span>
              <span className="text-xs font-bold tracking-wider">LIVE</span>
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-4 bg-midnight-800/40 p-3 rounded-2xl border border-white/5 backdrop-blur-sm">
          <button
            disabled={!modelReady || isRunning}
            onClick={start}
            className="flex-1 sm:flex-none btn-primary"
          >
            <Camera className="w-4 h-4" /> Start Feed
          </button>
          <button
            disabled={!isRunning}
            onClick={stop}
            className="flex-1 sm:flex-none btn-danger"
          >
            <CameraOff className="w-4 h-4" /> Stop Feed
          </button>

          <div className="w-px h-8 bg-white/10 hidden sm:block mx-2"></div>

          <button
            onClick={() => setAlertOn(v => !v)}
            className={`flex-1 sm:flex-none flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 border ${alertOn
                ? 'bg-amber-500/10 text-amber-400 border-amber-500/20 hover:bg-amber-500/20'
                : 'bg-transparent text-slate-500 border-white/5 hover:bg-white/5'
              }`}
          >
            {alertOn ? <Bell className="w-4 h-4" /> : <BellOff className="w-4 h-4" />}
            {alertOn ? 'Alerts Enabled' : 'Alerts Muted'}
          </button>
        </div>
      </div>

      {/* Right panel */}
      <div className="space-y-6">
        <CompliancePanel result={result} />
        <ViolationLog log={violations} onClear={() => setViolations([])} />
      </div>
    </div>
  )
}
