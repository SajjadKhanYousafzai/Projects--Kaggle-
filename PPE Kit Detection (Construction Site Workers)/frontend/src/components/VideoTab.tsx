import React, { useRef, useState, useEffect, useCallback } from 'react'
import { detect } from '../utils/detector'
import { drawDetections } from '../utils/drawing'
import { ComplianceResult, ViolationEntry } from '../types'
import { CompliancePanel } from './CompliancePanel'
import { ViolationLog } from './ViolationLog'
import { Video, Play, Pause, Upload } from 'lucide-react'

interface Props {
  modelReady: boolean
  confThresh: number
}

export const VideoTab: React.FC<Props> = ({ modelReady, confThresh }) => {
  const videoRef   = useRef<HTMLVideoElement>(null)
  const canvasRef  = useRef<HTMLCanvasElement>(null)
  const rafRef     = useRef<number>(0)
  const runningRef = useRef(false)
  const inputRef   = useRef<HTMLInputElement>(null)

  const [result,     setResult]     = useState<ComplianceResult | null>(null)
  const [violations, setViolations] = useState<ViolationEntry[]>([])
  const [isPlaying,  setIsPlaying]  = useState(false)
  const [videoSrc,   setVideoSrc]   = useState<string | null>(null)
  const lastViolRef = useRef(0)

  const loop = useCallback(async () => {
    if (!runningRef.current) return
    const video  = videoRef.current
    const canvas = canvasRef.current
    if (video && canvas && !video.paused && !video.ended) {
      try {
        const res = await detect(video, confThresh)
        drawDetections(canvas, video, res.detections)
        setResult(res)

        if (!res.compliant && Date.now() - lastViolRef.current > 3000) {
          lastViolRef.current = Date.now()
          setViolations(prev => [
            ...prev,
            {
              id:        prev.length + 1,
              timestamp: `${video.currentTime.toFixed(1)}s`,
              missing:   res.missing,
            },
          ])
        }
      } catch (_) { /* single-frame errors are fine */ }
    }
    rafRef.current = requestAnimationFrame(loop)
  }, [confThresh])

  const onFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (videoSrc) URL.revokeObjectURL(videoSrc)
    setVideoSrc(URL.createObjectURL(file))
    setViolations([])
    setResult(null)
    setIsPlaying(false)
  }

  const togglePlay = () => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) {
      video.play()
      runningRef.current = true
      setIsPlaying(true)
      rafRef.current = requestAnimationFrame(loop)
    } else {
      video.pause()
      runningRef.current = false
      cancelAnimationFrame(rafRef.current)
      setIsPlaying(false)
    }
  }

  useEffect(() => () => {
    runningRef.current = false
    cancelAnimationFrame(rafRef.current)
  }, [])

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2 space-y-4">
        <div className="relative bg-black rounded-xl overflow-hidden min-h-80 flex items-center justify-center">
          {videoSrc ? (
            <>
              <video
                ref={videoRef}
                src={videoSrc}
                className="absolute inset-0 w-full h-full object-contain opacity-0 pointer-events-none"
                onEnded={() => { runningRef.current = false; setIsPlaying(false) }}
              />
              <canvas
                ref={canvasRef}
                className="max-w-full max-h-[60vh] object-contain"
              />
            </>
          ) : (
            <div
              className="flex flex-col items-center gap-3 text-gray-500 p-8 cursor-pointer"
              onClick={() => inputRef.current?.click()}
            >
              <Video className="w-12 h-12" />
              <p className="text-sm">Click to load a video file</p>
            </div>
          )}
        </div>
        <input ref={inputRef} type="file" accept="video/*" className="hidden" onChange={onFile} />
        <div className="flex items-center gap-3">
          <button
            disabled={!modelReady}
            onClick={() => inputRef.current?.click()}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 rounded-lg text-sm font-medium transition-colors"
          >
            <Upload className="w-4 h-4" /> Load Video
          </button>
          <button
            disabled={!modelReady || !videoSrc}
            onClick={togglePlay}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-500 disabled:opacity-40 rounded-lg text-sm font-medium transition-colors"
          >
            {isPlaying ? <><Pause className="w-4 h-4" /> Pause</> : <><Play className="w-4 h-4" /> Play & Detect</>}
          </button>
        </div>
      </div>

      <div className="space-y-4">
        <CompliancePanel result={result} />
        <ViolationLog log={violations} onClear={() => setViolations([])} />
      </div>
    </div>
  )
}
