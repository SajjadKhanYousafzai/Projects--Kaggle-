// Import WASM-only bundle — avoids JSEP/WebGPU dynamic .mjs imports that break Vite
import * as ort from 'onnxruntime-web/wasm'
import {
  CLASS_NAMES,
  classType,
  Detection,
  ComplianceResult,
  POSITIVE_CLASSES,
  NEGATIVE_CLASSES,
} from '../types'

// ── Configuration ──────────────────────────────────────────────────────────────
const MODEL_PATH  = '/models/best.onnx'
const INPUT_SIZE  = 640
const CONF_THRESH = 0.35
const IOU_THRESH  = 0.45

let session: ort.InferenceSession | null = null
let isLoading = false

// ── Load model (singleton) ────────────────────────────────────────────────────
export async function loadModel(
  onProgress?: (msg: string) => void,
): Promise<void> {
  if (session || isLoading) return
  isLoading = true
  try {
    onProgress?.('Configuring ONNX runtime…')
    // Serve WASM files locally from /public (avoids CDN CORS + blob: worker issues)
    ort.env.wasm.wasmPaths = '/'
    // Single-threaded mode — avoids "Failed to fetch blob:" SharedArrayBuffer worker error
    ort.env.wasm.numThreads = 1
    onProgress?.('Fetching model weights (best.onnx)…')
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    })
    onProgress?.('Model ready ✅')
  } finally {
    isLoading = false
  }
}

export function isModelLoaded(): boolean {
  return session !== null
}

// ── Pre-processing ────────────────────────────────────────────────────────────
function preprocess(
  source: HTMLCanvasElement | HTMLVideoElement | HTMLImageElement,
): { tensor: ort.Tensor; scale: number; padX: number; padY: number } {
  const canvas = document.createElement('canvas')
  canvas.width  = INPUT_SIZE
  canvas.height = INPUT_SIZE
  const ctx = canvas.getContext('2d')!

  const srcW = source instanceof HTMLVideoElement ? source.videoWidth  : source.width  as number
  const srcH = source instanceof HTMLVideoElement ? source.videoHeight : source.height as number

  // Letterbox: scale to fit 640×640 keeping aspect ratio
  const scale = Math.min(INPUT_SIZE / srcW, INPUT_SIZE / srcH)
  const newW  = Math.round(srcW * scale)
  const newH  = Math.round(srcH * scale)
  const padX  = (INPUT_SIZE - newW) / 2
  const padY  = (INPUT_SIZE - newH) / 2

  ctx.fillStyle = '#808080'
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE)
  ctx.drawImage(source, padX, padY, newW, newH)

  const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data
  const float32   = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE)

  // HWC → CHW, normalise to [0, 1]
  for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
    float32[i]                             = imageData[i * 4]     / 255 // R
    float32[INPUT_SIZE * INPUT_SIZE + i]   = imageData[i * 4 + 1] / 255 // G
    float32[2 * INPUT_SIZE * INPUT_SIZE + i] = imageData[i * 4 + 2] / 255 // B
  }

  const tensor = new ort.Tensor('float32', float32, [1, 3, INPUT_SIZE, INPUT_SIZE])
  return { tensor, scale, padX, padY }
}

// ── NMS ───────────────────────────────────────────────────────────────────────
function iou(a: Detection, b: Detection): number {
  const ix1 = Math.max(a.x1, b.x1)
  const iy1 = Math.max(a.y1, b.y1)
  const ix2 = Math.min(a.x2, b.x2)
  const iy2 = Math.min(a.y2, b.y2)
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1)
  const aArea = (a.x2 - a.x1) * (a.y2 - a.y1)
  const bArea = (b.x2 - b.x1) * (b.y2 - b.y1)
  return inter / (aArea + bArea - inter + 1e-6)
}

function nms(dets: Detection[], iouThresh: number): Detection[] {
  dets.sort((a, b) => b.confidence - a.confidence)
  const kept: Detection[] = []
  const suppressed = new Set<number>()
  for (let i = 0; i < dets.length; i++) {
    if (suppressed.has(i)) continue
    kept.push(dets[i])
    for (let j = i + 1; j < dets.length; j++) {
      if (!suppressed.has(j) && iou(dets[i], dets[j]) > iouThresh) {
        suppressed.add(j)
      }
    }
  }
  return kept
}

// ── Post-processing ───────────────────────────────────────────────────────────
// YOLO11s ONNX output shape: [1, 15, 8400]
//   dim-1 0..3  → cx, cy, w, h  (in 640px space)
//   dim-1 4..14 → class raw logits (11 classes)
function postprocess(
  output: ort.Tensor,
  confThresh: number,
  iouThresh: number,
  scale: number,
  padX: number,
  padY: number,
  srcW: number,
  srcH: number,
): Detection[] {
  // data layout: [batch=1, 15, 8400]  → access [cls_or_box, anchor]
  const data    = output.data as Float32Array
  const numCls  = CLASS_NAMES.length        // 11
  const numAnch = 8400
  const candidates: Detection[] = []

  for (let a = 0; a < numAnch; a++) {
    // Get class scores — use sigmoid for raw logits
    let maxScore = -Infinity
    let maxCls   = 0
    for (let c = 0; c < numCls; c++) {
      const idx   = (4 + c) * numAnch + a
      const score = 1 / (1 + Math.exp(-data[idx]))  // sigmoid
      if (score > maxScore) { maxScore = score; maxCls = c }
    }
    if (maxScore < confThresh) continue

    // Box in 640px letterbox space
    const cx = data[0 * numAnch + a]
    const cy = data[1 * numAnch + a]
    const bw = data[2 * numAnch + a]
    const bh = data[3 * numAnch + a]

    // Remove letterbox padding and scale back to original image size
    const x1 = ((cx - bw / 2) - padX) / scale
    const y1 = ((cy - bh / 2) - padY) / scale
    const x2 = ((cx + bw / 2) - padX) / scale
    const y2 = ((cy + bh / 2) - padY) / scale

    candidates.push({
      x1: Math.max(0, x1),
      y1: Math.max(0, y1),
      x2: Math.min(srcW, x2),
      y2: Math.min(srcH, y2),
      classId:    maxCls,
      className:  CLASS_NAMES[maxCls],
      confidence: maxScore,
      type:       classType(CLASS_NAMES[maxCls]),
    })
  }

  return nms(candidates, iouThresh)
}

// ── Main inference entry point ────────────────────────────────────────────────
export async function detect(
  source: HTMLCanvasElement | HTMLVideoElement | HTMLImageElement,
  confThresh = CONF_THRESH,
): Promise<ComplianceResult> {
  if (!session) throw new Error('Model not loaded. Call loadModel() first.')

  const srcW = source instanceof HTMLVideoElement ? source.videoWidth  : source.width  as number
  const srcH = source instanceof HTMLVideoElement ? source.videoHeight : source.height as number

  const { tensor, scale, padX, padY } = preprocess(source)

  const t0      = performance.now()
  const feeds   = { [session.inputNames[0]]: tensor }
  const results = await session.run(feeds)
  const fps     = 1000 / (performance.now() - t0)

  const output     = results[session.outputNames[0]]
  const detections = postprocess(output, confThresh, IOU_THRESH, scale, padX, padY, srcW, srcH)

  const wearing = [...new Set(
    detections.filter(d => POSITIVE_CLASSES.has(d.className)).map(d => d.className),
  )]
  const missing = [...new Set(
    detections.filter(d => NEGATIVE_CLASSES.has(d.className)).map(d => d.className),
  )]

  return {
    wearing,
    missing,
    compliant:  missing.length === 0,
    detections,
    fps: Math.round(fps),
  }
}
