import React, { useRef, useState, useCallback, useEffect } from 'react'
import { detect } from '../utils/detector'
import { drawDetections } from '../utils/drawing'
import { ComplianceResult } from '../types'
import { CompliancePanel } from './CompliancePanel'
import { Upload, ImageIcon, MousePointerClick } from 'lucide-react'

interface Props {
  modelReady: boolean
  confThresh: number
}

export const ImageTab: React.FC<Props> = ({ modelReady, confThresh }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const [result, setResult] = useState<ComplianceResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [imgSrc, setImgSrc] = useState<string | null>(null)
  const [isHover, setIsHover] = useState(false)

  const run = useCallback(async (src: string) => {
    const img = new Image()
    img.onload = async () => {
      if (!canvasRef.current) return
      setLoading(true)
      try {
        const res = await detect(img, confThresh)
        drawDetections(canvasRef.current!, img, res.detections)
        setResult(res)
      } finally {
        setLoading(false)
      }
    }
    img.src = src
  }, [confThresh])

  // Re-run detection whenever confThresh changes (debounced 300ms)
  useEffect(() => {
    if (!imgSrc) return
    const timer = setTimeout(() => run(imgSrc), 300)
    return () => clearTimeout(timer)
  }, [confThresh]) // eslint-disable-line react-hooks/exhaustive-deps

  const onFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const url = URL.createObjectURL(file)
    setImgSrc(url)
    run(url)
    e.target.value = ''  // reset so same file can be re-uploaded
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsHover(false)
    const file = e.dataTransfer.files[0]
    if (!file || !file.type.startsWith('image/')) return
    const url = URL.createObjectURL(file)
    setImgSrc(url)
    run(url)
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      <div className="lg:col-span-2 space-y-6">
        {/* Drop zone / canvas */}
        <div
          className={`relative bg-midnight-900 rounded-2xl overflow-hidden min-h-[400px] flex items-center justify-center border-2 border-dashed transition-all duration-300 cursor-pointer shadow-xl ${isHover
              ? 'border-emerald-500 bg-emerald-500/5 scale-[1.01] shadow-[0_0_30px_rgba(16,185,129,0.1)]'
              : 'border-white/10 hover:border-emerald-500/50 hover:bg-white/5'
            }`}
          onDrop={onDrop}
          onDragOver={e => { e.preventDefault(); setIsHover(true) }}
          onDragLeave={() => setIsHover(false)}
          onClick={() => inputRef.current?.click()}
        >
          {imgSrc ? (
            <canvas
              ref={canvasRef}
              className="max-w-full max-h-[60vh] object-contain"
            />
          ) : (
            <div className="flex flex-col items-center gap-4 p-8 text-center animate-fade-in">
              <div className="w-20 h-20 rounded-full bg-midnight-800 flex items-center justify-center border border-white/5 relative shadow-inner mb-2">
                <ImageIcon className="w-10 h-10 text-emerald-500/80" />
                <MousePointerClick className="w-5 h-5 text-teal-400 absolute bottom-4 right-4 animate-bounce" />
              </div>
              <h3 className="text-xl font-semibold text-slate-200">
                Upload Target Image
              </h3>
              <p className="text-slate-400 text-sm max-w-xs">
                Drag and drop a high-resolution image here, or click to browse your files
              </p>
              <div className="flex gap-2 mt-2">
                {['JPG', 'PNG', 'WEBP'].map(ext => (
                  <span key={ext} className="text-[10px] font-bold tracking-wider px-2 py-1 rounded bg-white/5 text-slate-500 border border-white/5">
                    {ext}
                  </span>
                ))}
              </div>
            </div>
          )}
          {loading && (
            <div className="absolute inset-0 bg-midnight-900/80 backdrop-blur-sm flex flex-col items-center justify-center gap-4 z-10 transition-opacity">
              <div className="w-12 h-12 rounded-full border-4 border-emerald-500/20 border-t-emerald-500 animate-spin"></div>
              <span className="text-emerald-400 font-medium tracking-wide animate-pulse">Running YOLO Inference...</span>
            </div>
          )}
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={onFile}
        />

        <div className="flex justify-center sm:justify-start">
          <button
            disabled={!modelReady}
            onClick={() => inputRef.current?.click()}
            className="w-full sm:w-auto btn-primary"
          >
            <Upload className="w-4 h-4" />
            {imgSrc ? 'Upload New Image' : 'Select Image'}
          </button>
        </div>
      </div>

      <div>
        <CompliancePanel result={result} loading={loading} />
      </div>
    </div>
  )
}
