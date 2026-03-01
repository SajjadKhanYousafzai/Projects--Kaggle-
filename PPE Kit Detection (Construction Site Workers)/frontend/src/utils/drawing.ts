import { Detection, CLASS_COLORS, classType } from '../types'

export function drawDetections(
  canvas: HTMLCanvasElement,
  source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
  detections: Detection[],
): void {
  const ctx = canvas.getContext('2d')!

  // Mirror canvas size to source
  const srcW = source instanceof HTMLVideoElement ? source.videoWidth  : (source as HTMLImageElement).naturalWidth  || source.width
  const srcH = source instanceof HTMLVideoElement ? source.videoHeight : (source as HTMLImageElement).naturalHeight || source.height
  canvas.width  = srcW
  canvas.height = srcH

  ctx.drawImage(source, 0, 0, srcW, srcH)

  for (const det of detections) {
    const color = CLASS_COLORS[det.className] ?? '#808080'
    const { x1, y1, x2, y2 } = det
    const bw = x2 - x1
    const bh = y2 - y1

    // Box
    ctx.strokeStyle = color
    ctx.lineWidth   = Math.max(2, Math.min(srcW, srcH) * 0.003)
    ctx.strokeRect(x1, y1, bw, bh)

    // Label background
    const label    = `${det.className}  ${(det.confidence * 100).toFixed(0)}%`
    const fontSize = Math.max(12, Math.min(srcW, srcH) * 0.02)
    ctx.font       = `bold ${fontSize}px Inter, system-ui, sans-serif`
    const metrics  = ctx.measureText(label)
    const lw       = metrics.width + 8
    const lh       = fontSize + 8
    const ly       = y1 - lh < 0 ? y1 + lh : y1

    ctx.fillStyle = color
    ctx.fillRect(x1, ly - lh, lw, lh)

    // Label text
    ctx.fillStyle = '#ffffff'
    ctx.fillText(label, x1 + 4, ly - 4)

    // Violation overlay tint for negative classes
    if (classType(det.className) === 'negative') {
      ctx.fillStyle = 'rgba(220,38,38,0.10)'
      ctx.fillRect(x1, y1, bw, bh)
    }
  }
}
