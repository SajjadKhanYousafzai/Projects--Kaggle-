import React, { useEffect, useState } from 'react'
import { loadModel } from './utils/detector'
import { Header } from './components/Header'
import { WebcamTab } from './components/WebcamTab'
import { ImageTab } from './components/ImageTab'
import { VideoTab } from './components/VideoTab'
import { ClassLegend } from './components/ClassLegend'
import { TabId } from './types'
import { Camera, Image as ImageIcon, Video, SlidersHorizontal, Settings2, HardHat } from 'lucide-react'

const TABS: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: 'webcam', label: 'Live Camera', icon: <Camera className="w-4 h-4" /> },
  { id: 'image', label: 'Image Upload', icon: <ImageIcon className="w-4 h-4" /> },
  { id: 'video', label: 'Video File', icon: <Video className="w-4 h-4" /> },
]

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('webcam')
  const [modelReady, setModelReady] = useState(false)
  const [modelStatus, setModelStatus] = useState('Initializing engine…')
  const [confThresh, setConfThresh] = useState(0.35)
  const [showSettings, setShowSettings] = useState(false)

  useEffect(() => {
    loadModel(msg => setModelStatus(msg))
      .then(() => { setModelReady(true); setModelStatus('System Ready') })
      .catch(e => setModelStatus(`Error: ${e.message}`))
  }, [])

  return (
    <div className="min-h-screen flex flex-col font-sans">
      <Header modelReady={modelReady} modelStatus={modelStatus} />

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 py-8 space-y-8 relative z-10 animate-fade-in">
        {/* Navigation & Controls */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 glass-panel p-2 rounded-2xl">
          <nav className="flex w-full sm:w-auto p-1 bg-midnight-900/50 rounded-xl gap-1 overflow-x-auto no-scrollbar">
            {TABS.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 sm:flex-none flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold transition-all duration-300 relative overflow-hidden ${activeTab === tab.id
                  ? 'text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
                  }`}
              >
                {activeTab === tab.id && (
                  <div className="absolute inset-0 bg-gradient-to-r from-emerald-600/90 to-teal-600/90 mix-blend-multiply"></div>
                )}
                <span className="relative z-10 flex items-center gap-2">
                  {tab.icon}
                  {tab.label}
                </span>
                {activeTab === tab.id && (
                  <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-12 h-0.5 bg-white blur-sm"></div>
                )}
              </button>
            ))}
          </nav>

          <button
            onClick={() => setShowSettings(v => !v)}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all ${showSettings
              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
              : 'bg-midnight-900/50 hover:bg-midnight-700 text-slate-300 border border-transparent'
              }`}
          >
            <Settings2 className={`w-4 h-4 transition-transform duration-300 ${showSettings ? 'rotate-90' : ''}`} />
            Settings
          </button>
        </div>

        {/* Floating Settings Panel */}
        <div
          className={`transition-all duration-500 origin-top overflow-hidden ${showSettings ? 'max-h-40 opacity-100 mb-6' : 'max-h-0 opacity-0 mb-0'
            }`}
        >
          <div className="glass-panel p-6 rounded-2xl flex flex-wrap items-center justify-between gap-6 border-l-4 border-l-emerald-500">
            <div className="flex flex-col gap-1 w-full max-w-md">
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm font-medium text-slate-300 flex items-center gap-2">
                  <SlidersHorizontal className="w-4 h-4 text-emerald-400" />
                  Confidence Threshold
                </label>
                <span className="text-sm font-bold bg-midnight-900 px-2.5 py-1 rounded-md text-emerald-400 border border-white/5">
                  {(confThresh * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min={0.1}
                max={0.9}
                step={0.05}
                value={confThresh}
                onChange={e => setConfThresh(Number(e.target.value))}
                className="w-full h-2 bg-midnight-900 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
            </div>

            <div className="flex gap-4 text-xs font-mono text-slate-400 bg-midnight-900/50 p-3 rounded-xl border border-white/5">
              <div className="flex flex-col">
                <span className="text-slate-500 mb-1 font-sans">Engine</span>
                <span className="text-emerald-200">YOLO11s ONNX</span>
              </div>
              <div className="w-px bg-white/10"></div>
              <div className="flex flex-col">
                <span className="text-slate-500 mb-1 font-sans">Classes</span>
                <span className="text-emerald-200">11 Targets</span>
              </div>
              <div className="w-px bg-white/10"></div>
              <div className="flex flex-col">
                <span className="text-slate-500 mb-1 font-sans">Resolution</span>
                <span className="text-emerald-200">640x640</span>
              </div>
            </div>
          </div>
        </div>

        {/* Interactive Workspace */}
        <div className="relative">
          <div className={`transition-opacity duration-300 ${activeTab === 'webcam' ? 'block animate-slide-up' : 'hidden'}`}>
            <WebcamTab modelReady={modelReady} confThresh={confThresh} />
          </div>
          <div className={`transition-opacity duration-300 ${activeTab === 'image' ? 'block animate-slide-up' : 'hidden'}`}>
            <ImageTab modelReady={modelReady} confThresh={confThresh} />
          </div>
          <div className={`transition-opacity duration-300 ${activeTab === 'video' ? 'block animate-slide-up' : 'hidden'}`}>
            <VideoTab modelReady={modelReady} confThresh={confThresh} />
          </div>
        </div>

        {/* Global Legend */}
        <div className="animate-slide-up" style={{ animationDelay: '0.2s' }}>
          <ClassLegend />
        </div>
      </main>

      <footer className="mt-auto py-6 text-center">
        <div className="inline-flex items-center justify-center gap-3 px-6 py-2 rounded-full glass-panel text-xs text-slate-400">
          <span className="flex items-center gap-1.5"><HardHat className="w-3.5 h-3.5" /> PPE Kit Detection</span>
          <span className="w-1 h-1 rounded-full bg-slate-600"></span>
          <span>YOLO11s</span>
          <span className="w-1 h-1 rounded-full bg-slate-600"></span>
          <span>React + TS</span>
        </div>
      </footer>
    </div>
  )
}
