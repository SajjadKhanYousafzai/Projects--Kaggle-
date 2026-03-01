// ── Class definitions matching the trained YOLO11s model ──────────────────────

export const CLASS_NAMES: string[] = [
  'Helmet',     // 0
  'Gloves',     // 1
  'Vest',       // 2
  'Boots',      // 3
  'Goggles',    // 4
  'none',       // 5
  'Person',     // 6
  'no_helmet',  // 7
  'no_goggle',  // 8
  'no_gloves',  // 9
  'no_boots',   // 10
]

export const POSITIVE_CLASSES = new Set([
  'Helmet', 'Gloves', 'Vest', 'Boots', 'Goggles', 'Person',
])

export const NEGATIVE_CLASSES = new Set([
  'no_helmet', 'no_goggle', 'no_gloves', 'no_boots',
])

// Colour per class [R, G, B]
export const CLASS_COLORS: Record<string, string> = {
  Helmet:     '#00c800',
  Gloves:     '#0096ff',
  Vest:       '#ffc800',
  Boots:      '#00ffff',
  Goggles:    '#c800ff',
  Person:     '#6464ff',
  none:       '#b4b4b4',
  no_helmet:  '#ff0000',
  no_goggle:  '#c80064',
  no_gloves:  '#ff5000',
  no_boots:   '#960000',
}

export type ClassType = 'positive' | 'negative' | 'neutral'

export function classType(name: string): ClassType {
  if (POSITIVE_CLASSES.has(name)) return 'positive'
  if (NEGATIVE_CLASSES.has(name)) return 'negative'
  return 'neutral'
}

// ── Detection box ─────────────────────────────────────────────────────────────

export interface Detection {
  x1: number
  y1: number
  x2: number
  y2: number
  classId: number
  className: string
  confidence: number
  type: ClassType
}

// ── Compliance summary ────────────────────────────────────────────────────────

export interface ComplianceResult {
  wearing:   string[]
  missing:   string[]
  compliant: boolean
  detections: Detection[]
  fps?: number
}

// ── Violation log entry ───────────────────────────────────────────────────────

export interface ViolationEntry {
  id:        number
  timestamp: string
  missing:   string[]
  snapshot?: string   // base64 data URL
}

// ── App tab ───────────────────────────────────────────────────────────────────

export type TabId = 'webcam' | 'image' | 'video'
