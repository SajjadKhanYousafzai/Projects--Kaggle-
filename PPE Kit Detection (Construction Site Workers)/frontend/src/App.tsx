import React, { useEffect, useState } from 'react'
import { loadModel } from './utils/detector'
import { Header } from './components/Header'
import { WebcamTab } from './components/WebcamTab'
import { ImageTab } from './components/ImageTab'
import { VideoTab } from './components/VideoTab'
import { ClassLegend } from './components/ClassLegend'
import { TabId } from './types'
import { Camera, Image as ImageIcon, Video, SlidersHorizontal } from 'lucide-react'

const TABS: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: 'webcam', label: 'Live Camera',   icon: <Camera className="w-4 h-4" /> },
  { id: 'image',  label: 'Image Upload',  icon: <ImageIcon className="w-4 h-4" /> },
  { id: 'video',  label: 'Video File',    icon: <Video className="w-4 h-4" /> },
]

export default function App() {
  const [activeTab,    setActiveTab]    = useState<TabId>('webcam')
  const [modelReady,   setModelReady]   = useState(false)
  const [modelStatus,  setModelStatus]  = useState('Loading model…')
  const [confThresh,   setConfThresh]   = useState(0.35)
  const [showSettings, setShowSettings] = useState(false)

  useEffect(() => {
    loadModel(msg => setModelStatus(msg))
      .then(() => { setModelReady(true); setModelStatus('Model ready') })
      .catch(e  => setModelStatus(`Error: ${e.message}`))
  }, [])

  return (
    <div className="min-h-screen bg-gray-950 flex flex-col">
      <Header modelReady={modelReady} modelStatus={modelStatus} />

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 py-6 space-y-6">
        {/* Tab bar + settings */}
        <div className="flex items-center justify-between flex-wrap gap-3">
          <nav className="flex bg-gray-900 rounded-xl p-1 gap-1">
            {TABS.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'bg-yellow-500 text-gray-950'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800'
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </nav>

          <button
            onClick={() => setShowSettings(v => !v)}
            className="flex items-center gap-2 px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm text-gray-300 transition-colors"
          >
            <SlidersHorizontal className="w-4 h-4" />
            Settings
          </button>
        </div>

        {/* Settings panel */}
        {showSettings && (
          <div className="bg-gray-900 rounded-xl p-5 flex items-center gap-6 flex-wrap">
            <div className="flex items-center gap-3">
              <label className="text-sm text-gray-400 whitespace-nowrap">
                Confidence Threshold
              </label>
              <input
                type="range"
                min={0.1}
                max={0.9}
                step={0.05}
                value={confThresh}
                onChange={e => setConfThresh(Number(e.target.value))}
                className="w-32 accent-yellow-400"
              />
              <span className="text-sm font-mono text-yellow-400 w-10">
                {(confThresh * 100).toFixed(0)}%
              </span>
            </div>
            <div className="text-xs text-gray-500">
              Model: YOLO11s · mAP50: 53.9% · 11 classes · imgsz: 640
            </div>
          </div>
        )}

        {/* Active tab content */}
        {activeTab === 'webcam' && (
          <WebcamTab modelReady={modelReady} confThresh={confThresh} />
        )}
        {activeTab === 'image' && (
          <ImageTab modelReady={modelReady} confThresh={confThresh} />
        )}
        {activeTab === 'video' && (
          <VideoTab modelReady={modelReady} confThresh={confThresh} />
        )}

        {/* Class legend always visible */}
        <ClassLegend />
      </main>

      <footer className="border-t border-gray-800 py-3 text-center text-xs text-gray-600">
        PPE Kit Detection · YOLO11s · ONNX Runtime Web · React + TypeScript
      </footer>
    </div>
  )
}
