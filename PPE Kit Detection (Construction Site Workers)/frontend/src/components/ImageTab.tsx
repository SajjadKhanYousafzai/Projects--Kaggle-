import React, { useRef, useState, useCallback } from 'react'
import { detect } from '../utils/detector'
import { drawDetections } from '../utils/drawing'
import { ComplianceResult } from '../types'
import { CompliancePanel } from './CompliancePanel'
import { Upload, ImageIcon } from 'lucide-react'

interface Props {
  modelReady: boolean
  confThresh: number
}

export const ImageTab: React.FC<Props> = ({ modelReady, confThresh }) => {
  const canvasRef  = useRef<HTMLCanvasElement>(null)
  const imgRef     = useRef<HTMLImageElement>(null)
  const inputRef   = useRef<HTMLInputElement>(null)

  const [result,   setResult]  = useState<ComplianceResult | null>(null)
  const [loading,  setLoading] = useState(false)
  const [imgSrc,   setImgSrc]  = useState<string | null>(null)

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

  const onFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const url = URL.createObjectURL(file)
    setImgSrc(url)
    run(url)
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (!file || !file.type.startsWith('image/')) return
    const url = URL.createObjectURL(file)
    setImgSrc(url)
    run(url)
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2 space-y-4">
        {/* Drop zone / canvas */}
        <div
          className="relative bg-black rounded-xl overflow-hidden min-h-80 flex items-center justify-center border-2 border-dashed border-gray-700 hover:border-yellow-500 transition-colors cursor-pointer"
          onDrop={onDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => inputRef.current?.click()}
        >
          {imgSrc ? (
            <canvas
              ref={canvasRef}
              className="max-w-full max-h-[60vh] object-contain"
            />
          ) : (
            <div className="flex flex-col items-center gap-3 text-gray-500 p-8">
              <ImageIcon className="w-12 h-12" />
              <p className="text-sm font-medium">Drop an image or click to browse</p>
              <p className="text-xs">JPG · PNG · WEBP</p>
            </div>
          )}
          {loading && (
            <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
              <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-yellow-400" />
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
        <button
          disabled={!modelReady}
          onClick={() => inputRef.current?.click()}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed rounded-lg text-sm font-medium transition-colors"
        >
          <Upload className="w-4 h-4" /> Upload Image
        </button>
      </div>

      <div>
        <CompliancePanel result={result} loading={loading} />
      </div>
    </div>
  )
}
