import React, { useRef, useState, useEffect, useCallback } from 'react'
import { detect } from '../utils/detector'
import { drawDetections } from '../utils/drawing'
import { ComplianceResult, ViolationEntry } from '../types'
import { CompliancePanel } from './CompliancePanel'
import { ViolationLog } from './ViolationLog'
import { Video, Play, Pause, Upload, FileVideo } from 'lucide-react'

interface Props {
  modelReady: boolean
  confThresh: number
}

export const VideoTab: React.FC<Props> = ({ modelReady, confThresh }) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number>(0)
  const runningRef = useRef(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const [result, setResult] = useState<ComplianceResult | null>(null)
  const [violations, setViolations] = useState<ViolationEntry[]>([])
  const [isPlaying, setIsPlaying] = useState(false)
  const [videoSrc, setVideoSrc] = useState<string | null>(null)
  const [isHover, setIsHover] = useState(false)
  const lastViolRef = useRef(0)

  const loop = useCallback(async () => {
    if (!runningRef.current) return
    const video = videoRef.current
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
              id: prev.length + 1,
              timestamp: `${video.currentTime.toFixed(1)}s`,
              missing: res.missing,
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
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      <div className="lg:col-span-2 space-y-6">
        <div
          className={`relative bg-midnight-900 rounded-2xl overflow-hidden min-h-[400px] flex items-center justify-center shadow-xl border-2 border-dashed transition-all duration-300 ${videoSrc
              ? 'border-white/10'
              : isHover
                ? 'border-emerald-500 bg-emerald-500/5 cursor-pointer scale-[1.01] shadow-[0_0_30px_rgba(16,185,129,0.1)]'
                : 'border-white/10 hover:border-emerald-500/50 hover:bg-white/5 cursor-pointer'
            }`}
          onDragOver={e => { e.preventDefault(); if (!videoSrc) setIsHover(true); }}
          onDragLeave={() => setIsHover(false)}
          onDrop={e => {
            e.preventDefault()
            setIsHover(false)
            if (videoSrc) return
            const file = e.dataTransfer.files[0]
            if (file && file.type.startsWith('video/')) {
              if (videoSrc) URL.revokeObjectURL(videoSrc)
              setVideoSrc(URL.createObjectURL(file))
              setViolations([])
              setResult(null)
              setIsPlaying(false)
            }
          }}
          onClick={() => { if (!videoSrc) inputRef.current?.click() }}
        >
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
                className="max-w-full max-h-[60vh] object-contain relative z-10"
              />

              {!isPlaying && (
                <div className="absolute inset-0 bg-midnight-900/60 backdrop-blur-sm flex items-center justify-center z-20">
                  <button
                    onClick={(e) => { e.stopPropagation(); togglePlay(); }}
                    className="w-20 h-20 rounded-full bg-emerald-500/90 text-white flex items-center justify-center shadow-[0_0_30px_rgba(16,185,129,0.5)] hover:bg-emerald-400 hover:scale-110 transition-all duration-300"
                  >
                    <Play className="w-10 h-10 ml-2" />
                  </button>
                </div>
              )}
            </>
          ) : (
            <div className="flex flex-col items-center gap-4 p-8 text-center animate-fade-in">
              <div className="w-20 h-20 rounded-full bg-midnight-800 flex items-center justify-center border border-white/5 relative shadow-inner mb-2">
                <FileVideo className="w-10 h-10 text-emerald-500/80" />
              </div>
              <h3 className="text-xl font-semibold text-slate-200">
                Load Target Video
              </h3>
              <p className="text-slate-400 text-sm max-w-xs">
                Drag and drop a video file here, or click to browse your files
              </p>
              <div className="flex gap-2 mt-2">
                {['MP4', 'WEBM', 'MOV'].map(ext => (
                  <span key={ext} className="text-[10px] font-bold tracking-wider px-2 py-1 rounded bg-white/5 text-slate-500 border border-white/5">
                    {ext}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        <input ref={inputRef} type="file" accept="video/*" className="hidden" onChange={onFile} />

        <div className="flex flex-wrap items-center gap-4 bg-midnight-800/40 p-3 rounded-2xl border border-white/5 backdrop-blur-sm">
          <button
            disabled={!modelReady}
            onClick={() => inputRef.current?.click()}
            className="flex-1 sm:flex-none btn-secondary border-none shadow-none"
          >
            <Upload className="w-4 h-4" />
            {videoSrc ? 'Change Video' : 'Load Video'}
          </button>

          {videoSrc && <div className="w-px h-8 bg-white/10 hidden sm:block mx-2"></div>}

          <button
            disabled={!modelReady || !videoSrc}
            onClick={togglePlay}
            className={`flex-1 sm:flex-none ${isPlaying ? 'btn-danger' : 'btn-primary'}`}
          >
            {isPlaying ? <><Pause className="w-4 h-4" /> Pause Detection</> : <><Play className="w-4 h-4" /> Play & Detect</>}
          </button>
        </div>
      </div>

      <div className="space-y-6">
        <CompliancePanel result={result} />
        <ViolationLog log={violations} onClear={() => setViolations([])} />
      </div>
    </div>
  )
}
