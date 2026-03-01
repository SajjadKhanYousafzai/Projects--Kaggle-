import { classType, Detection, ComplianceResult } from '../types'

// ── Configuration ──────────────────────────────────────────────────────────────
// VITE_API_URL is set in .env (local) or .env.production (deployed)
const API_BASE    = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:7860'
const CONF_THRESH = 0.35

let _ready    = false
let _isLoading = false

// ── Connect to backend (replaces ONNX model load) ─────────────────────────────
export async function loadModel(
  onProgress?: (msg: string) => void,
): Promise<void> {
  if (_ready || _isLoading) return
  _isLoading = true
  try {
    onProgress?.('Connecting to PPE detection API…')
    const res = await fetch(`${API_BASE}/health`)
    if (!res.ok) throw new Error(`API responded with ${res.status}`)
    const data = await res.json()
    if (!data.model_loaded) throw new Error('Backend model is still loading — please retry.')
    _ready = true
    onProgress?.('API connected ✅')
  } finally {
    _isLoading = false
  }
}

export function isModelLoaded(): boolean {
  return _ready
}

// ── Convert canvas / video / image element → JPEG Blob ────────────────────────
function toBlob(
  source: HTMLCanvasElement | HTMLVideoElement | HTMLImageElement,
): Promise<Blob> {
  const canvas = document.createElement('canvas')
  if (source instanceof HTMLVideoElement) {
    canvas.width  = source.videoWidth
    canvas.height = source.videoHeight
  } else {
    canvas.width  = (source as HTMLCanvasElement | HTMLImageElement).width  as number
    canvas.height = (source as HTMLCanvasElement | HTMLImageElement).height as number
  }
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(source, 0, 0)
  return new Promise((resolve, reject) =>
    canvas.toBlob(
      blob => (blob ? resolve(blob) : reject(new Error('toBlob failed'))),
      'image/jpeg',
      0.92,
    ),
  )
}

// ── Main inference entry point ────────────────────────────────────────────────
export async function detect(
  source: HTMLCanvasElement | HTMLVideoElement | HTMLImageElement,
  confThresh = CONF_THRESH,
): Promise<ComplianceResult> {
  if (!_ready) throw new Error('API not connected. Call loadModel() first.')

  const blob = await toBlob(source)
  const form = new FormData()
  form.append('file', blob, 'frame.jpg')

  const t0  = performance.now()
  const res = await fetch(
    `${API_BASE}/predict?conf=${confThresh}`,
    { method: 'POST', body: form },
  )
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`API error ${res.status}: ${text}`)
  }
  const fps  = Math.round(1000 / (performance.now() - t0))
  const data = await res.json()

  const detections: Detection[] = (data.detections as Array<{
    class_id: number
    class_name: string
    confidence: number
    bbox: [number, number, number, number]
  }>).map(d => ({
    x1:         d.bbox[0],
    y1:         d.bbox[1],
    x2:         d.bbox[2],
    y2:         d.bbox[3],
    classId:    d.class_id,
    className:  d.class_name,
    confidence: d.confidence,
    type:       classType(d.class_name),
  }))

  return {
    wearing:    data.compliance.wearing   as string[],
    missing:    data.compliance.missing   as string[],
    compliant:  data.compliance.is_compliant as boolean,
    detections,
    fps,
  }
}
