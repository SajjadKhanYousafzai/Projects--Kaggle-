"use client";

import React, { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, PieChart, Pie, RadarChart,
  Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Legend, LineChart, Line, Area, AreaChart
} from "recharts";
import {
  Activity, Shield, Target, Zap, Microscope, AlertTriangle,
  TrendingUp, Database, Heart, Brain, FlaskConical, Stethoscope,
  ChevronDown, ExternalLink, Github
} from "lucide-react";

/* ================================================================
   DATA — Hard-coded from notebook training results
   ================================================================ */
const CELL_TYPE_DATA = [
  { name: "Neutrophil", count: 1100, anomaly: 0, category: "Normal_WBC" },
  { name: "Lymphocyte", count: 850, anomaly: 0, category: "Normal_WBC" },
  { name: "Normal_RBC", count: 900, anomaly: 0, category: "Normal_RBC" },
  { name: "Monocyte", count: 400, anomaly: 0, category: "Normal_WBC" },
  { name: "Platelet", count: 300, anomaly: 0, category: "Normal_Platelet" },
  { name: "Eosinophil", count: 300, anomaly: 0, category: "Normal_WBC" },
  { name: "Blast_Cell", count: 280, anomaly: 1, category: "Leukemia" },
  { name: "Elliptocyte", count: 200, anomaly: 1, category: "Anemia" },
  { name: "Prolymphocyte", count: 180, anomaly: 1, category: "Leukemia" },
  { name: "Schistocyte", count: 170, anomaly: 1, category: "Anemia" },
  { name: "Hyper_Neutrophil", count: 160, anomaly: 1, category: "Infection" },
  { name: "Basophil", count: 150, anomaly: 0, category: "Normal_WBC" },
  { name: "Spherocyte", count: 150, anomaly: 1, category: "Anemia" },
  { name: "React_Lymphocyte", count: 150, anomaly: 1, category: "Infection" },
  { name: "Sickle_Cell", count: 140, anomaly: 1, category: "Sickle_Cell" },
  { name: "Toxic_Gran", count: 140, anomaly: 1, category: "Infection" },
  { name: "Target_Cell", count: 130, anomaly: 1, category: "Anemia" },
  { name: "Smudge_Cell", count: 100, anomaly: 1, category: "Artefact" },
  { name: "Artefact", count: 80, anomaly: 1, category: "Artefact" },
].sort((a, b) => b.count - a.count);

const MODEL_DATA = [
  { name: "CatBoost", accuracy: 0.981, auc: 0.9983, recall: 0.9688, f1: 0.9697 },
  { name: "XGBoost", accuracy: 0.980, auc: 0.9981, recall: 0.9664, f1: 0.9678 },
  { name: "LightGBM", accuracy: 0.9795, auc: 0.9979, recall: 0.964, f1: 0.9661 },
  { name: "RandomForest", accuracy: 0.978, auc: 0.9975, recall: 0.9616, f1: 0.9638 },
  { name: "LogRegression", accuracy: 0.972, auc: 0.9949, recall: 0.9568, f1: 0.9541 },
];

const FEATURE_IMPORTANCE = [
  { name: "Chromatin Density", value: 0.1245 },
  { name: "Nucleus Area %", value: 0.118 },
  { name: "Granularity Score", value: 0.1034 },
  { name: "Cytoplasm Ratio", value: 0.0987 },
  { name: "Lobularity Score", value: 0.0923 },
  { name: "Cell Diameter", value: 0.0891 },
  { name: "Eccentricity", value: 0.0834 },
  { name: "Membrane Smooth.", value: 0.0776 },
  { name: "Circularity", value: 0.0712 },
  { name: "Cell Area (px)", value: 0.0456 },
].sort((a, b) => a.value - b.value);

const BENCHMARK_DATA = [
  { task: "Anomaly Detection (AUC)", cytodiffusion: 0.99, ours: 0.998, baseline: 0.916 },
  { task: "Standard Classification", cytodiffusion: 0.985, ours: 0.936, baseline: 0.738 },
  { task: "Low-Data Accuracy", cytodiffusion: 0.962, ours: 0.94, baseline: 0.924 },
  { task: "Blast Detection Sens.", cytodiffusion: 0.905, ours: 0.969, baseline: null },
];

const REFERENCE_TABLE = [
  { cell: "Neutrophil", disease: "Normal_WBC", anomaly: 0, significance: "Primary bacterial infection fighter; most common WBC", count: 1100 },
  { cell: "Lymphocyte", disease: "Normal_WBC", anomaly: 0, significance: "Adaptive immunity; elevated in viral infections", count: 850 },
  { cell: "Normal_RBC", disease: "Normal_RBC", anomaly: 0, significance: "Oxygen transport; biconcave disc shape", count: 900 },
  { cell: "Monocyte", disease: "Normal_WBC", anomaly: 0, significance: "Phagocyte; precursor to macrophages", count: 400 },
  { cell: "Eosinophil", disease: "Normal_WBC", anomaly: 0, significance: "Allergic response & parasite defense", count: 300 },
  { cell: "Basophil", disease: "Normal_WBC", anomaly: 0, significance: "Least common WBC; allergic reactions", count: 150 },
  { cell: "Platelet", disease: "Normal_Platelet", anomaly: 0, significance: "Blood clotting; anucleate cell fragment", count: 300 },
  { cell: "Blast_Cell", disease: "Leukemia", anomaly: 1, significance: "CRITICAL — immature cell; hallmark of leukemia", count: 280 },
  { cell: "Prolymphocyte", disease: "Leukemia", anomaly: 1, significance: "Immature lymphocyte; found in CLL/PLL", count: 180 },
  { cell: "Elliptocyte", disease: "Anemia", anomaly: 1, significance: "Elongated RBC; iron deficiency marker", count: 200 },
  { cell: "Schistocyte", disease: "Anemia", anomaly: 1, significance: "Fragmented RBC; hemolytic anemia / TTP", count: 170 },
  { cell: "Spherocyte", disease: "Anemia", anomaly: 1, significance: "Spherical RBC; hereditary spherocytosis", count: 150 },
  { cell: "Sickle_Cell", disease: "Sickle_Cell", anomaly: 1, significance: "Crescent RBC; HbS mutation", count: 140 },
  { cell: "Target_Cell", disease: "Anemia", anomaly: 1, significance: "Central dense RBC; thalassemia / liver disease", count: 130 },
  { cell: "Hyper_Neutrophil", disease: "Infection", anomaly: 1, significance: "5+ lobes; B12/folate deficiency marker", count: 160 },
  { cell: "Toxic_Granulation", disease: "Infection", anomaly: 1, significance: "Heavy dark granules; severe bacterial sepsis", count: 140 },
  { cell: "React_Lymphocyte", disease: "Infection", anomaly: 1, significance: "Enlarged atypical lymph; EBV/CMV infection", count: 150 },
  { cell: "Smudge_Cell", disease: "Artefact", anomaly: 1, significance: "Ruptured lymphocyte; CLL hallmark", count: 100 },
  { cell: "Artefact", disease: "Artefact", anomaly: 1, significance: "Preparation artefact; exclude from analysis", count: 80 },
];

const DISEASE_PIE = [
  { name: "Normal_WBC", value: 2800, color: "#10B981" },
  { name: "Normal_RBC", value: 900, color: "#34D399" },
  { name: "Normal_Platelet", value: 300, color: "#6EE7B7" },
  { name: "Leukemia", value: 460, color: "#EF4444" },
  { name: "Anemia", value: 650, color: "#F59E0B" },
  { name: "Sickle_Cell", value: 140, color: "#F97316" },
  { name: "Infection", value: 450, color: "#8B5CF6" },
  { name: "Artefact", value: 180, color: "#6B7280" },
];

const RADAR_DATA = [
  { metric: "Accuracy", CatBoost: 98.1, XGBoost: 98.0, RandomForest: 97.8 },
  { metric: "AUC-ROC", CatBoost: 99.8, XGBoost: 99.8, RandomForest: 99.7 },
  { metric: "Recall", CatBoost: 96.9, XGBoost: 96.6, RandomForest: 96.2 },
  { metric: "Precision", CatBoost: 97.1, XGBoost: 96.9, RandomForest: 96.6 },
  { metric: "F1-Score", CatBoost: 97.0, XGBoost: 96.8, RandomForest: 96.4 },
];

/* ================================================================
   CUSTOM TOOLTIP
   ================================================================ */
interface TooltipPayload {
  name?: string;
  value?: number;
  color?: string;
  dataKey?: string;
  payload?: Record<string, unknown>;
}

const CustomTooltip = ({ active, payload, label }: {
  active?: boolean;
  payload?: TooltipPayload[];
  label?: string;
}) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'rgba(17,24,39,0.95)', border: '1px solid rgba(99,102,241,0.3)',
      borderRadius: 12, padding: '12px 16px', backdropFilter: 'blur(12px)',
      boxShadow: '0 8px 32px rgba(0,0,0,0.4)'
    }}>
      <p style={{ color: '#9CA3AF', fontSize: 12, marginBottom: 6 }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color || '#F9FAFB', fontSize: 13, fontWeight: 600 }}>
          {p.name || p.dataKey}: {typeof p.value === 'number' && p.value < 1
            ? (p.value * 100).toFixed(2) + '%'
            : p.value?.toLocaleString()}
        </p>
      ))}
    </div>
  );
};

/* ================================================================
   KPI CARD
   ================================================================ */
const KpiCard = ({ icon: Icon, label, value, subtitle, color, delay }: {
  icon: React.ElementType; label: string; value: string;
  subtitle: string; color: string; delay: number;
}) => (
  <div className="animate-fade-in-up" style={{
    animationDelay: `${delay}s`,
    background: 'var(--bg-card)', borderRadius: 16,
    padding: '28px 24px', border: '1px solid var(--border-subtle)',
    position: 'relative', overflow: 'hidden',
    transition: 'all 0.3s ease',
    cursor: 'default',
  }}
    onMouseEnter={e => {
      (e.currentTarget as HTMLDivElement).style.border = `1px solid ${color}40`;
      (e.currentTarget as HTMLDivElement).style.boxShadow = `0 0 30px ${color}15`;
      (e.currentTarget as HTMLDivElement).style.transform = 'translateY(-2px)';
    }}
    onMouseLeave={e => {
      (e.currentTarget as HTMLDivElement).style.border = '1px solid var(--border-subtle)';
      (e.currentTarget as HTMLDivElement).style.boxShadow = 'none';
      (e.currentTarget as HTMLDivElement).style.transform = 'translateY(0)';
    }}
  >
    <div style={{
      position: 'absolute', top: -20, right: -20,
      width: 80, height: 80, borderRadius: '50%',
      background: `${color}08`,
    }} />
    <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
      <div style={{
        width: 40, height: 40, borderRadius: 10,
        background: `${color}15`, display: 'flex',
        alignItems: 'center', justifyContent: 'center',
      }}>
        <Icon size={20} color={color} />
      </div>
      <span style={{ color: 'var(--text-secondary)', fontSize: 13, fontWeight: 500 }}>{label}</span>
    </div>
    <div style={{ fontSize: 36, fontWeight: 800, color, letterSpacing: '-0.02em', lineHeight: 1 }}>
      {value}
    </div>
    <div style={{ color: 'var(--text-muted)', fontSize: 12, marginTop: 8 }}>{subtitle}</div>
  </div>
);

/* ================================================================
   SECTION WRAPPER
   ================================================================ */
const Section = ({ title, subtitle, children, id }: {
  title: string; subtitle: string; children: React.ReactNode; id?: string;
}) => (
  <section id={id} style={{ marginBottom: 64 }} className="animate-fade-in-up">
    <div style={{ marginBottom: 28 }}>
      <h2 style={{
        fontSize: 24, fontWeight: 700, color: 'var(--text-primary)',
        marginBottom: 6,
      }}>{title}</h2>
      <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>{subtitle}</p>
    </div>
    {children}
  </section>
);

/* ================================================================
   PREDICTION DEMO
   ================================================================ */
const FEATURE_RANGES: Record<string, { min: number; max: number; default: number; unit: string }> = {
  cell_diameter_um: { min: 4, max: 25, default: 12, unit: "μm" },
  nucleus_area_pct: { min: 5, max: 80, default: 35, unit: "%" },
  chromatin_density: { min: 0.1, max: 1, default: 0.5, unit: "" },
  cytoplasm_ratio: { min: 0.1, max: 0.9, default: 0.45, unit: "" },
  circularity: { min: 0.3, max: 1, default: 0.85, unit: "" },
  eccentricity: { min: 0, max: 0.95, default: 0.3, unit: "" },
  granularity_score: { min: 0, max: 1, default: 0.25, unit: "" },
  lobularity_score: { min: 0, max: 1, default: 0.15, unit: "" },
  membrane_smoothness: { min: 0, max: 1, default: 0.8, unit: "" },
};

const PredictionDemo = () => {
  const [features, setFeatures] = useState<Record<string, number>>(
    Object.fromEntries(Object.entries(FEATURE_RANGES).map(([k, v]) => [k, v.default]))
  );
  const [result, setResult] = useState<{
    label: string; confidence: number; risk: string; probability: number;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState(false);

  const predict = async () => {
    setLoading(true);
    setApiError(false);
    try {
      const res = await fetch("http://localhost:8000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(features),
      });
      if (!res.ok) throw new Error("API error");
      const data = await res.json();
      setResult({
        label: data.label,
        confidence: data.confidence,
        risk: data.risk_level,
        probability: data.anomaly_probability,
      });
    } catch {
      // Fallback: client-side mock prediction
      setApiError(true);
      const score = (
        features.chromatin_density * 0.3 +
        features.nucleus_area_pct / 80 * 0.25 +
        features.granularity_score * 0.2 +
        features.lobularity_score * 0.15 +
        (1 - features.circularity) * 0.1
      );
      const prob = Math.min(Math.max(score, 0.02), 0.98);
      setResult({
        label: prob > 0.5 ? "Anomaly" : "Normal",
        confidence: Math.max(prob, 1 - prob),
        risk: prob < 0.3 ? "Low" : prob < 0.6 ? "Moderate" : prob < 0.85 ? "High" : "Critical",
        probability: prob,
      });
    }
    setLoading(false);
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32 }}>
      {/* Sliders */}
      <div style={{
        background: 'var(--bg-card)', borderRadius: 16,
        padding: 28, border: '1px solid var(--border-subtle)',
      }}>
        <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 20, color: 'var(--text-primary)' }}>
          Cell Morphology Features
        </h3>
        {Object.entries(FEATURE_RANGES).map(([key, range]) => (
          <div key={key} style={{ marginBottom: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <label style={{ fontSize: 12, color: 'var(--text-secondary)', textTransform: 'capitalize' }}>
                {key.replace(/_/g, ' ')}
              </label>
              <span style={{ fontSize: 12, color: 'var(--accent-cyan)', fontWeight: 600 }}>
                {features[key].toFixed(2)}{range.unit}
              </span>
            </div>
            <input
              type="range"
              min={range.min}
              max={range.max}
              step={(range.max - range.min) / 100}
              value={features[key]}
              onChange={e => setFeatures(prev => ({ ...prev, [key]: parseFloat(e.target.value) }))}
              style={{
                width: '100%', height: 4, borderRadius: 2,
                appearance: 'none', background: '#374151', outline: 'none',
                accentColor: 'var(--accent-indigo)',
              }}
            />
          </div>
        ))}
        <button
          onClick={predict}
          disabled={loading}
          style={{
            width: '100%', padding: '14px 0', borderRadius: 12,
            background: loading ? '#374151' : 'linear-gradient(135deg, #6366F1, #8B5CF6)',
            color: 'white', border: 'none', fontSize: 15, fontWeight: 700,
            cursor: loading ? 'wait' : 'pointer', marginTop: 8,
            transition: 'all 0.3s ease',
            boxShadow: loading ? 'none' : '0 4px 20px rgba(99,102,241,0.3)',
          }}
        >
          {loading ? "Analyzing..." : "🔬 Run Prediction"}
        </button>
      </div>

      {/* Result */}
      <div style={{
        background: 'var(--bg-card)', borderRadius: 16,
        padding: 28, border: '1px solid var(--border-subtle)',
        display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center',
        textAlign: 'center',
      }}>
        {!result ? (
          <div style={{ color: 'var(--text-muted)' }}>
            <Microscope size={64} style={{ opacity: 0.3, marginBottom: 16 }} />
            <p style={{ fontSize: 15 }}>Adjust the cell features and click<br /><strong>Run Prediction</strong> to see results</p>
          </div>
        ) : (
          <div className="animate-fade-in-up">
            {apiError && (
              <div style={{
                background: 'rgba(245,158,11,0.1)', border: '1px solid rgba(245,158,11,0.3)',
                borderRadius: 8, padding: '8px 12px', marginBottom: 16, fontSize: 11,
                color: '#F59E0B',
              }}>
                Backend offline — using client-side estimation
              </div>
            )}
            <div style={{
              width: 120, height: 120, borderRadius: '50%',
              background: result.label === 'Normal'
                ? 'linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05))'
                : 'linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05))',
              border: `3px solid ${result.label === 'Normal' ? '#10B981' : '#EF4444'}`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              margin: '0 auto 20px',
              boxShadow: result.label === 'Normal'
                ? '0 0 40px rgba(16,185,129,0.2)' : '0 0 40px rgba(239,68,68,0.2)',
            }}>
              {result.label === 'Normal'
                ? <Shield size={48} color="#10B981" />
                : <AlertTriangle size={48} color="#EF4444" />}
            </div>
            <div style={{
              fontSize: 28, fontWeight: 800,
              color: result.label === 'Normal' ? '#10B981' : '#EF4444',
              marginBottom: 8,
            }}>
              {result.label}
            </div>
            <div style={{
              display: 'inline-block',
              padding: '4px 16px', borderRadius: 20,
              background: result.risk === 'Low' ? 'rgba(16,185,129,0.15)'
                : result.risk === 'Moderate' ? 'rgba(245,158,11,0.15)'
                  : result.risk === 'High' ? 'rgba(249,115,22,0.15)'
                    : 'rgba(239,68,68,0.15)',
              color: result.risk === 'Low' ? '#10B981'
                : result.risk === 'Moderate' ? '#F59E0B'
                  : result.risk === 'High' ? '#F97316' : '#EF4444',
              fontSize: 13, fontWeight: 600, marginBottom: 20,
            }}>
              {result.risk} Risk
            </div>
            <div style={{
              display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16, width: '100%',
            }}>
              <div style={{
                background: 'var(--bg-secondary)', borderRadius: 12, padding: 16,
              }}>
                <div style={{ color: 'var(--text-muted)', fontSize: 11, marginBottom: 4 }}>Confidence</div>
                <div style={{ fontSize: 24, fontWeight: 800, color: 'var(--accent-cyan)' }}>
                  {(result.confidence * 100).toFixed(1)}%
                </div>
              </div>
              <div style={{
                background: 'var(--bg-secondary)', borderRadius: 12, padding: 16,
              }}>
                <div style={{ color: 'var(--text-muted)', fontSize: 11, marginBottom: 4 }}>Anomaly Prob.</div>
                <div style={{ fontSize: 24, fontWeight: 800, color: 'var(--accent-purple)' }}>
                  {(result.probability * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

/* ================================================================
   MAIN PAGE
   ================================================================ */
export default function Home() {
  const totalNormal = CELL_TYPE_DATA.filter(c => c.anomaly === 0).reduce((s, c) => s + c.count, 0);
  const totalAnomaly = CELL_TYPE_DATA.filter(c => c.anomaly === 1).reduce((s, c) => s + c.count, 0);

  return (
    <main style={{ maxWidth: 1280, margin: '0 auto', padding: '0 24px' }}>
      {/* ── HERO ─────────────────────────────────────────── */}
      <header style={{
        padding: '80px 0 60px',
        textAlign: 'center',
        position: 'relative',
      }}>
        <div style={{
          position: 'absolute', top: 0, left: '50%', transform: 'translateX(-50%)',
          width: 600, height: 600, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%)',
          pointerEvents: 'none',
        }} />
        <div className="animate-fade-in-up" style={{ position: 'relative' }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 8,
            padding: '6px 16px', borderRadius: 20,
            background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.2)',
            fontSize: 13, color: 'var(--accent-indigo)', fontWeight: 500,
            marginBottom: 20,
          }}>
            <FlaskConical size={14} /> Machine Learning Research
          </div>
          <h1 style={{
            fontSize: 52, fontWeight: 900, lineHeight: 1.1,
            background: 'linear-gradient(135deg, #F9FAFB 0%, #6366F1 50%, #8B5CF6 100%)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
            marginBottom: 16, letterSpacing: '-0.03em',
          }}>
            Blood Cell Anomaly<br />Detection
          </h1>
          <p style={{
            fontSize: 18, color: 'var(--text-secondary)', maxWidth: 640,
            margin: '0 auto 32px', lineHeight: 1.7,
          }}>
            AI-powered hematology analysis detecting 19 cell types across 8 disease categories
            using ensemble machine learning with <span style={{ color: '#10B981', fontWeight: 600 }}>98.1% accuracy</span> and <span style={{ color: '#EF4444', fontWeight: 600 }}>96.9% anomaly recall</span>.
          </p>
          <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
            <a href="#prediction" style={{
              padding: '12px 28px', borderRadius: 12,
              background: 'linear-gradient(135deg, #6366F1, #8B5CF6)',
              color: 'white', textDecoration: 'none', fontWeight: 600, fontSize: 14,
              boxShadow: '0 4px 20px rgba(99,102,241,0.3)',
              transition: 'all 0.3s ease',
            }}>
              Try Live Demo →
            </a>
            <a href="#models" style={{
              padding: '12px 28px', borderRadius: 12,
              background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-subtle)',
              color: 'var(--text-primary)', textDecoration: 'none', fontWeight: 600, fontSize: 14,
              transition: 'all 0.3s ease',
            }}>
              View Results
            </a>
          </div>
        </div>
      </header>

      {/* ── KPI CARDS ────────────────────────────────────── */}
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 20, marginBottom: 64,
      }}>
        <KpiCard icon={Target} label="Accuracy" value="98.1%" subtitle="Best model (CatBoost)" color="#6366F1" delay={0.1} />
        <KpiCard icon={TrendingUp} label="AUC-ROC" value="0.9983" subtitle="Near-perfect discrimination" color="#10B981" delay={0.2} />
        <KpiCard icon={Shield} label="Anomaly Recall" value="96.9%" subtitle="Catches 97 of 100 anomalies" color="#EF4444" delay={0.3} />
        <KpiCard icon={Database} label="Dataset Size" value="5,880" subtitle="19 cell types • 36 features" color="#F59E0B" delay={0.4} />
      </div>

      {/* ── CELL TYPE DISTRIBUTION ────────────────────────── */}
      <Section title="Cell Type Distribution" subtitle="19 cell types color-coded by anomaly status — Green = Normal, Red = Anomaly">
        <div style={{
          background: 'var(--bg-card)', borderRadius: 16,
          padding: 28, border: '1px solid var(--border-subtle)',
        }}>
          <ResponsiveContainer width="100%" height={520}>
            <BarChart data={CELL_TYPE_DATA} layout="vertical" margin={{ left: 120, right: 40, top: 10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1F2937" horizontal={false} />
              <XAxis type="number" stroke="#6B7280" fontSize={11} />
              <YAxis type="category" dataKey="name" stroke="#9CA3AF" fontSize={11} width={110} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="count" radius={[0, 6, 6, 0]} barSize={18}>
                {CELL_TYPE_DATA.map((entry, index) => (
                  <Cell key={index} fill={entry.anomaly === 0 ? '#10B981' : '#EF4444'} fillOpacity={0.85} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div style={{
            display: 'flex', gap: 24, justifyContent: 'center', marginTop: 16,
            color: 'var(--text-secondary)', fontSize: 13,
          }}>
            <span>🟢 Normal cells: <strong style={{ color: '#10B981' }}>{totalNormal.toLocaleString()}</strong></span>
            <span>🔴 Anomaly cells: <strong style={{ color: '#EF4444' }}>{totalAnomaly.toLocaleString()}</strong></span>
            <span>Anomaly rate: <strong style={{ color: '#F59E0B' }}>{(totalAnomaly / (totalNormal + totalAnomaly) * 100).toFixed(1)}%</strong></span>
          </div>
        </div>
      </Section>

      {/* ── DISEASE CATEGORY PIE ─────────────────────────── */}
      <Section title="Disease Category Breakdown" subtitle="Distribution of samples across 8 clinical disease categories">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
          <div style={{
            background: 'var(--bg-card)', borderRadius: 16,
            padding: 28, border: '1px solid var(--border-subtle)',
          }}>
            <ResponsiveContainer width="100%" height={350}>
              <PieChart>
                <Pie
                  data={DISEASE_PIE}
                  cx="50%" cy="50%"
                  innerRadius={80} outerRadius={140}
                  paddingAngle={3} dataKey="value"
                  stroke="none"
                >
                  {DISEASE_PIE.map((entry, index) => (
                    <Cell key={index} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend
                  formatter={(value) => <span style={{ color: '#9CA3AF', fontSize: 12 }}>{value}</span>}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div style={{
            background: 'var(--bg-card)', borderRadius: 16,
            padding: 28, border: '1px solid var(--border-subtle)',
          }}>
            <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 16 }}>Category Details</h3>
            {DISEASE_PIE.map((d) => (
              <div key={d.name} style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '10px 0', borderBottom: '1px solid var(--border-subtle)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div style={{ width: 10, height: 10, borderRadius: '50%', background: d.color }} />
                  <span style={{ fontSize: 13 }}>{d.name}</span>
                </div>
                <div style={{ display: 'flex', gap: 16 }}>
                  <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{d.value.toLocaleString()}</span>
                  <span style={{ fontSize: 13, color: 'var(--accent-cyan)', fontWeight: 600, minWidth: 45, textAlign: 'right' }}>
                    {(d.value / 5880 * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* ── MODEL COMPARISON ─────────────────────────────── */}
      <Section id="models" title="Model Performance Comparison" subtitle="5 binary classifiers compared across accuracy, AUC-ROC, recall, and F1-score">
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 24 }}>
          <div style={{
            background: 'var(--bg-card)', borderRadius: 16,
            padding: 28, border: '1px solid var(--border-subtle)',
          }}>
            <ResponsiveContainer width="100%" height={380}>
              <BarChart data={MODEL_DATA} margin={{ top: 20, right: 20, bottom: 10, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1F2937" />
                <XAxis dataKey="name" stroke="#9CA3AF" fontSize={11} />
                <YAxis domain={[0.94, 1]} stroke="#6B7280" fontSize={11} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                <Tooltip content={<CustomTooltip />} />
                <Legend formatter={(value) => <span style={{ color: '#9CA3AF', fontSize: 12 }}>{value}</span>} />
                <Bar dataKey="accuracy" name="Accuracy" fill="#6366F1" radius={[4, 4, 0, 0]} barSize={14} />
                <Bar dataKey="auc" name="AUC-ROC" fill="#10B981" radius={[4, 4, 0, 0]} barSize={14} />
                <Bar dataKey="recall" name="Recall" fill="#EF4444" radius={[4, 4, 0, 0]} barSize={14} />
                <Bar dataKey="f1" name="F1-Score" fill="#F59E0B" radius={[4, 4, 0, 0]} barSize={14} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{
            background: 'var(--bg-card)', borderRadius: 16,
            padding: 28, border: '1px solid var(--border-subtle)',
          }}>
            <ResponsiveContainer width="100%" height={380}>
              <RadarChart data={RADAR_DATA}>
                <PolarGrid stroke="#1F2937" />
                <PolarAngleAxis dataKey="metric" stroke="#9CA3AF" fontSize={11} />
                <PolarRadiusAxis domain={[94, 100]} stroke="#374151" fontSize={9} />
                <Radar name="CatBoost" dataKey="CatBoost" stroke="#6366F1" fill="#6366F1" fillOpacity={0.15} strokeWidth={2} />
                <Radar name="XGBoost" dataKey="XGBoost" stroke="#10B981" fill="#10B981" fillOpacity={0.1} strokeWidth={2} />
                <Radar name="RandomForest" dataKey="RandomForest" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.05} strokeWidth={2} />
                <Legend formatter={(value) => <span style={{ color: '#9CA3AF', fontSize: 11 }}>{value}</span>} />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </Section>

      {/* ── FEATURE IMPORTANCE ────────────────────────────── */}
      <Section title="Top Feature Importances" subtitle="Most influential morphological features for anomaly detection (Random Forest)">
        <div style={{
          background: 'var(--bg-card)', borderRadius: 16,
          padding: 28, border: '1px solid var(--border-subtle)',
        }}>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={FEATURE_IMPORTANCE} layout="vertical" margin={{ left: 140, right: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1F2937" horizontal={false} />
              <XAxis type="number" stroke="#6B7280" fontSize={11} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
              <YAxis type="category" dataKey="name" stroke="#9CA3AF" fontSize={12} width={130} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="value" name="Importance" radius={[0, 6, 6, 0]} barSize={20}>
                {FEATURE_IMPORTANCE.map((_, index) => (
                  <Cell key={index} fill={`hsl(${240 + index * 12}, 70%, ${55 + index * 3}%)`} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Section>

      {/* ── BENCHMARK ────────────────────────────────────── */}
      <Section title="Benchmark Comparison" subtitle="Our model vs CytoDiffusion (Nature Machine Intelligence 2025) and baselines">
        <div style={{
          background: 'var(--bg-card)', borderRadius: 16,
          padding: 28, border: '1px solid var(--border-subtle)',
        }}>
          <ResponsiveContainer width="100%" height={360}>
            <BarChart data={BENCHMARK_DATA} margin={{ top: 20, right: 30, bottom: 10, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1F2937" />
              <XAxis dataKey="task" stroke="#9CA3AF" fontSize={10} />
              <YAxis domain={[0.85, 1.02]} stroke="#6B7280" fontSize={11} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
              <Tooltip content={<CustomTooltip />} />
              <Legend formatter={(value) => <span style={{ color: '#9CA3AF', fontSize: 12 }}>{value}</span>} />
              <Bar dataKey="cytodiffusion" name="CytoDiffusion" fill="#F59E0B" radius={[4, 4, 0, 0]} barSize={22} />
              <Bar dataKey="ours" name="Our Model" fill="#10B981" radius={[4, 4, 0, 0]} barSize={22} />
              <Bar dataKey="baseline" name="Baseline" fill="#6B7280" radius={[4, 4, 0, 0]} barSize={22} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Section>

      {/* ── CLINICAL REFERENCE TABLE ──────────────────────── */}
      <Section title="Clinical Reference Table" subtitle="Complete cell type catalog with disease associations and clinical significance">
        <div style={{
          background: 'var(--bg-card)', borderRadius: 16,
          border: '1px solid var(--border-subtle)', overflow: 'hidden',
        }}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: 'rgba(99,102,241,0.08)' }}>
                  {['Cell Type', 'Disease Category', 'Status', 'Clinical Significance', 'Count'].map(h => (
                    <th key={h} style={{
                      padding: '14px 16px', textAlign: 'left',
                      fontWeight: 600, color: 'var(--text-secondary)',
                      borderBottom: '1px solid var(--border-subtle)',
                      whiteSpace: 'nowrap',
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {REFERENCE_TABLE.map((row, i) => (
                  <tr key={i} style={{
                    borderBottom: '1px solid var(--border-subtle)',
                    transition: 'background 0.2s',
                  }}
                    onMouseEnter={e => (e.currentTarget.style.background = 'rgba(99,102,241,0.04)')}
                    onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                  >
                    <td style={{ padding: '12px 16px', fontWeight: 600 }}>{row.cell}</td>
                    <td style={{ padding: '12px 16px', color: 'var(--text-secondary)' }}>{row.disease}</td>
                    <td style={{ padding: '12px 16px' }}>
                      <span style={{
                        padding: '3px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600,
                        background: row.anomaly === 0 ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)',
                        color: row.anomaly === 0 ? '#10B981' : '#EF4444',
                      }}>
                        {row.anomaly === 0 ? 'Normal' : 'Anomaly'}
                      </span>
                    </td>
                    <td style={{ padding: '12px 16px', color: 'var(--text-secondary)', maxWidth: 300 }}>{row.significance}</td>
                    <td style={{ padding: '12px 16px', fontWeight: 600, color: 'var(--accent-cyan)' }}>{row.count.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </Section>

      {/* ── PREDICTION DEMO ──────────────────────────────── */}
      <Section id="prediction" title="🔬 Live Prediction Demo" subtitle="Adjust cell morphology features and predict anomaly status in real-time">
        <PredictionDemo />
      </Section>

      {/* ── FOOTER ───────────────────────────────────────── */}
      <footer style={{
        borderTop: '1px solid var(--border-subtle)',
        padding: '32px 0', textAlign: 'center',
        color: 'var(--text-muted)', fontSize: 13,
        marginTop: 40,
      }}>
        <p>Blood Cell Anomaly Detection Dashboard • Built with Next.js, Recharts & FastAPI</p>
        <p style={{ marginTop: 6 }}>
          Model: CatBoost | Dataset: 5,880 samples | 19 Cell Types | 8 Disease Categories
        </p>
      </footer>
    </main>
  );
}
