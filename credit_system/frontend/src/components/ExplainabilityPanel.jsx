import { useMemo } from 'react'
import './ExplainabilityPanel.css'

/* ═══════════════════════════════════════════════════════════════════════════
   Prediction Gauge — SVG semicircle showing probability vs threshold
   ═══════════════════════════════════════════════════════════════════════════ */
function PredictionGauge({ probability, threshold, approved }) {
  const pct = Math.max(0, Math.min(1, probability))
  const thr = Math.max(0, Math.min(1, threshold))
  const r = 70, cx = 80, cy = 80, strokeW = 12
  const startAngle = Math.PI, endAngle = 2 * Math.PI

  const arc = (frac) => {
    const a = startAngle + frac * (endAngle - startAngle)
    return `${cx + r * Math.cos(a)} ${cy + r * Math.sin(a)}`
  }
  const pathD = (from, to) => {
    const a1 = startAngle + from * (endAngle - startAngle)
    const a2 = startAngle + to * (endAngle - startAngle)
    const x1 = cx + r * Math.cos(a1), y1 = cy + r * Math.sin(a1)
    const x2 = cx + r * Math.cos(a2), y2 = cy + r * Math.sin(a2)
    const large = to - from > 0.5 ? 1 : 0
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`
  }

  const thrAngle = startAngle + thr * (endAngle - startAngle)
  const thrX = cx + (r + 8) * Math.cos(thrAngle)
  const thrY = cy + (r + 8) * Math.sin(thrAngle)
  const thrX2 = cx + (r - 8) * Math.cos(thrAngle)
  const thrY2 = cy + (r - 8) * Math.sin(thrAngle)

  return (
    <svg className="xai-gauge-svg" width="160" height="95" viewBox="0 0 160 95">
      <path d={pathD(0, 1)} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={strokeW} strokeLinecap="round" />
      <path d={pathD(0, pct)} fill="none" stroke={approved ? '#34d399' : '#f87171'} strokeWidth={strokeW} strokeLinecap="round"
        style={{ filter: `drop-shadow(0 0 6px ${approved ? 'rgba(52,211,153,0.4)' : 'rgba(248,113,113,0.4)'})` }} />
      <line x1={thrX} y1={thrY} x2={thrX2} y2={thrY2} stroke="#f59e0b" strokeWidth="2.5" strokeLinecap="round" />
      <text x={cx} y={cy - 8} textAnchor="middle" fill="#0f172a" fontSize="22" fontWeight="700" fontFamily="JetBrains Mono, monospace">
        {(pct * 100).toFixed(0)}%
      </text>
      <text x={cx} y={cy + 10} textAnchor="middle" fill="#334155" fontSize="9">probability</text>
    </svg>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   Waterfall Chart — SHAP feature contributions cascading from base to output
   ═══════════════════════════════════════════════════════════════════════════ */
function WaterfallChart({ factors, baseValue }) {
  const sorted = useMemo(() =>
    [...factors].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)),
    [factors]
  )

  const steps = useMemo(() => {
    let running = baseValue ?? 0
    return sorted.map(f => {
      const start = running
      running += f.shap_value
      return { ...f, start, end: running }
    })
  }, [sorted, baseValue])

  const allVals = steps.flatMap(s => [s.start, s.end])
  allVals.push(baseValue ?? 0)
  const minVal = Math.min(...allVals), maxVal = Math.max(...allVals)
  const range = maxVal - minVal || 1
  const toX = (v) => ((v - minVal) / range) * 100
  const baseX = toX(baseValue ?? 0)

  return (
    <div className="xai-waterfall-container">
      {baseValue != null && (
        <div className="xai-wf-base-row">
          <span className="xai-wf-base-label">E[f(x)] base value</span>
          <span className="xai-wf-base-val">{baseValue.toFixed(4)}</span>
        </div>
      )}
      {steps.map((s, i) => {
        const left = Math.min(toX(s.start), toX(s.end))
        const width = Math.abs(toX(s.end) - toX(s.start))
        const isPos = s.shap_value >= 0
        return (
          <div className="xai-waterfall-row" key={s.feature + i}>
            <span className="xai-wf-label" title={s.label || s.feature}>{s.label || s.feature}</span>
            <div className="xai-wf-bar-area">
              <div className="xai-wf-baseline" style={{ left: `${baseX}%` }} />
              <div className={`xai-wf-bar ${isPos ? 'positive' : 'negative'}`}
                style={{ left: `${left}%`, width: `${Math.max(width, 0.5)}%` }} />
            </div>
            <span className={`xai-wf-value ${isPos ? 'positive' : 'negative'}`}>
              {isPos ? '+' : ''}{s.shap_value.toFixed(4)}
            </span>
          </div>
        )
      })}
      {baseValue != null && steps.length > 0 && (
        <div className="xai-wf-output-row">
          <span className="xai-wf-base-label">f(x) output</span>
          <span className="xai-wf-base-val">{steps[steps.length - 1].end.toFixed(4)}</span>
        </div>
      )}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   Feature Bars — Horizontal diverging bar chart centered at zero
   ═══════════════════════════════════════════════════════════════════════════ */
function FeatureBars({ factors }) {
  const sorted = useMemo(() =>
    [...factors].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)),
    [factors]
  )
  const maxAbs = Math.max(...sorted.map(f => Math.abs(f.shap_value)), 0.001)

  const fmt = (v) => {
    if (v == null) return ''
    if (typeof v === 'number') return v % 1 === 0 ? v.toLocaleString() : v.toFixed(2)
    return String(v)
  }

  return (
    <div className="xai-bars">
      {sorted.map((f, i) => {
        const isPos = f.shap_value >= 0
        const pct = (Math.abs(f.shap_value) / maxAbs) * 50
        const isRule = f.feature?.includes('_rule')
        return (
          <div className="xai-bar-row" key={f.feature + i}>
            <span className="xai-bar-name" title={f.label || f.feature}>
              {f.label || f.feature}
              {isRule && <span className="xai-tag xai-tag-rule" style={{ marginLeft: 4, fontSize: '0.55rem' }}>RULE</span>}
            </span>
            <div className="xai-bar-track">
              <div className="xai-bar-center" />
              <div className={`xai-bar-fill ${isPos ? 'positive' : 'negative'}`}
                style={{ width: `${pct}%` }} />
            </div>
            <span className={`xai-bar-val ${isPos ? 'positive' : 'negative'}`}>
              {isPos ? '+' : ''}{f.shap_value.toFixed(4)}
            </span>
            <span className="xai-bar-raw">{fmt(f.raw_value)}</span>
          </div>
        )
      })}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   Counterfactual Suggestions — What would change the decision
   ═══════════════════════════════════════════════════════════════════════════ */
function CounterfactualCards({ counterfactuals }) {
  if (!counterfactuals || counterfactuals.length === 0) return null
  return (
    <div className="xai-cf-grid">
      {counterfactuals.map((cf, i) => (
        <div className="xai-cf-card" key={cf.feature + i}>
          <div className="xai-cf-feature">{cf.label}</div>
          <div className="xai-cf-change">
            <span className="xai-cf-current">{cf.current_display}</span>
            <span className="xai-cf-arrow">→</span>
            <span className="xai-cf-target">{cf.target_display}</span>
          </div>
          {cf.tip && <div className="xai-cf-tip">{cf.tip}</div>}
        </div>
      ))}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   Decision Reasons — Plain-language explanation
   ═══════════════════════════════════════════════════════════════════════════ */
function DecisionReasons({ applicantMessage }) {
  if (!applicantMessage) return null
  const { summary, primary_reasons, actionable_tips } = applicantMessage
  return (
    <>
      {summary && <p style={{ color: '#0f172a', fontSize: '0.88rem', marginBottom: '0.75rem', fontWeight: '500' }}>{summary}</p>}
      {primary_reasons && primary_reasons.length > 0 && (
        <div className="xai-reasons-list">
          {primary_reasons.map((r, i) => (
            <div className="xai-reason" key={i}>
              <span className="xai-reason-icon">•</span>
              <span>{r}</span>
            </div>
          ))}
        </div>
      )}
      {actionable_tips && actionable_tips.length > 0 && (
        <div className="xai-tips-list">
          {actionable_tips.map((t, i) => <div className="xai-tip" key={i}>{t}</div>)}
        </div>
      )}
    </>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   Feature Contribution % — Pie-like percentage breakdown
   ═══════════════════════════════════════════════════════════════════════════ */
function ContributionBreakdown({ factors }) {
  const totalAbs = factors.reduce((s, f) => s + Math.abs(f.shap_value), 0) || 1
  const top = [...factors]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .slice(0, 5)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
      {top.map((f, i) => {
        const pct = (Math.abs(f.shap_value) / totalAbs * 100)
        const isPos = f.shap_value >= 0
        return (
          <div key={f.feature + i} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.78rem' }}>
            <span style={{ width: 140, minWidth: 140, textAlign: 'right', color: '#94a3b8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {f.label || f.feature}
            </span>
            <div style={{ flex: 1, height: 10, background: 'rgba(255,255,255,0.04)', borderRadius: 5, overflow: 'hidden' }}>
              <div style={{
                width: `${pct}%`, height: '100%', borderRadius: 5,
                background: isPos ? 'linear-gradient(90deg,#059669,#34d399)' : 'linear-gradient(90deg,#dc2626,#f87171)',
                transition: 'width 0.8s ease'
              }} />
            </div>
            <span style={{ width: 45, minWidth: 45, fontWeight: 600, fontSize: '0.75rem', color: isPos ? '#34d399' : '#f87171', fontFamily: 'JetBrains Mono, monospace' }}>
              {pct.toFixed(1)}%
            </span>
          </div>
        )
      })}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   Main ExplainabilityPanel
   ═══════════════════════════════════════════════════════════════════════════ */
export default function ExplainabilityPanel({
  shapFactors = [],
  probability,
  threshold,
  approved,
  confidenceLevel,
  baseValue,
  counterfactuals,
  modelVersion,
  applicantMessage,
  blockHash,
  liteMode = false,
}) {
  const hasShap = Array.isArray(shapFactors) && shapFactors.length > 0

  if (!hasShap) {
    return (
      <div className="xai-panel">
        <div className="xai-warning">
          ⚠️ SHAP explanations are not available for this evaluation. The explainability service may be temporarily unavailable.
        </div>
      </div>
    )
  }

  return (
    <div className={`xai-panel ${liteMode ? 'lite-mode' : ''}`}>

      {/* ── Prediction Gauge + Confidence ── */}
      {probability != null && (
        <div className="xai-section">
          <h4><span className="xai-icon">📊</span> Model Prediction <span className="xai-tag xai-tag-xai">XAI</span></h4>
          <div className="xai-gauge-wrap">
            <PredictionGauge probability={probability} threshold={threshold ?? 0.5} approved={approved} />
            <div className="xai-gauge-info">
              <span className={`xai-gauge-prob ${approved ? 'approved' : 'rejected'}`}>
                {approved ? '✅ You May Be Eligible' : '❌ You May Get Rejected'}
              </span>
              <span className="xai-gauge-label">
                Probability: <strong>{(probability * 100).toFixed(1)}%</strong>
              </span>
              <span className="xai-gauge-threshold">
                <span className="dot" /> Threshold: {((threshold ?? 0.5) * 100).toFixed(1)}%
              </span>
              {confidenceLevel && (
                <span className={`xai-gauge-confidence xai-conf-${confidenceLevel}`}>
                  {confidenceLevel} Confidence
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── SHAP Waterfall Chart ── */}
      {!liteMode && baseValue != null && (
        <div className="xai-section">
          <h4><span className="xai-icon">🌊</span> SHAP Waterfall <span className="xai-tag xai-tag-shap">SHAP</span></h4>
          <WaterfallChart factors={shapFactors} baseValue={baseValue} />
        </div>
      )}

      {/* ── Feature Importance Bars ── */}
      {!liteMode && (
        <div className="xai-section">
          <h4><span className="xai-icon">📈</span> Feature Impact Analysis <span className="xai-tag xai-tag-shap">SHAP</span></h4>
          <FeatureBars factors={shapFactors} />
        </div>
      )}

      {/* ── Contribution % Breakdown ── */}
      {!liteMode && (
        <div className="xai-section">
          <h4><span className="xai-icon">🎯</span> Contribution Breakdown</h4>
          <ContributionBreakdown factors={shapFactors} />
        </div>
      )}

      {/* ── Counterfactual Suggestions ── */}
      {!liteMode && counterfactuals && counterfactuals.length > 0 && (
        <div className="xai-section">
          <h4><span className="xai-icon">🔄</span> What Would Change The Decision?</h4>
          <CounterfactualCards counterfactuals={counterfactuals} />
        </div>
      )}

      {/* ── Decision Reasons ── */}
      {applicantMessage && (
        <div className="xai-section">
          <h4><span className="xai-icon">💬</span> Decision Explanation</h4>
          <DecisionReasons applicantMessage={applicantMessage} />
        </div>
      )}

      {/* ── Model Transparency ── */}
      {!liteMode && (
        <div className="xai-section">
          <h4><span className="xai-icon">🔍</span> Model Transparency</h4>
          <div className="xai-transparency-grid">
            {modelVersion && (
              <div className="xai-meta-item">
                <span className="xai-meta-label">Model Version</span>
                <span className="xai-meta-value">{modelVersion}</span>
              </div>
            )}
            <div className="xai-meta-item">
              <span className="xai-meta-label">Threshold</span>
              <span className="xai-meta-value">{((threshold ?? 0.5) * 100).toFixed(1)}%</span>
            </div>
            <div className="xai-meta-item">
              <span className="xai-meta-label">Factors Shown</span>
              <span className="xai-meta-value">{shapFactors.length}</span>
            </div>
            {blockHash && (
              <div className="xai-meta-item">
                <span className="xai-meta-label">Audit Hash</span>
                <span className="xai-meta-value" title={blockHash}>{blockHash.slice(0, 12)}…</span>
              </div>
            )}
            <div className="xai-meta-item">
              <span className="xai-meta-label">Explainer</span>
              <span className="xai-meta-value">SHAP TreeExplainer</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
