import { useState, useEffect } from 'react'
import './OfficerDashboard.css'
import { API } from '../config.js'

export default function OfficerDashboard() {
  const [decisions, setDecisions]   = useState([])
  const [chainStatus, setChainStatus] = useState(null)
  const [selected, setSelected]     = useState(null)
  const [loading, setLoading]       = useState(true)
  const [searchId, setSearchId]     = useState('')
  const [searchResult, setSearchResult] = useState(null)
  const [searchError, setSearchError]   = useState(null)

  // Load recent decisions + chain verification on mount
  useEffect(() => {
    async function load() {
      try {
        const [decRes, chainRes] = await Promise.all([
          fetch(`${API}/audit/recent?limit=30`),
          fetch(`${API}/audit/verify`),
        ])
        setDecisions(await decRes.json())
        setChainStatus(await chainRes.json())
      } catch (e) {
        console.error(e)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const handleSearch = async () => {
    if (!searchId.trim()) return
    setSearchError(null)
    setSearchResult(null)
    try {
      const res = await fetch(`${API}/audit/${searchId.trim()}`)
      if (!res.ok) throw new Error('Application ID not found')
      setSearchResult(await res.json())
      setSelected(null)
    } catch (e) {
      setSearchError(e.message)
    }
  }

  const detail = searchResult || selected

  return (
    <div className="od">
      {/* ── Header ─────────────────────────────── */}
      <div className="od-hero">
        <h2 className="od-title">Compliance Officer Dashboard</h2>
        <p className="od-desc">
          Review every AI decision with full SHAP attribution, audit chain verification,
          and anomaly detection.
        </p>
      </div>

      {/* ── Chain Status Banner ────────────────── */}
      {chainStatus && (
        <div className={`od-chain-banner ${chainStatus.valid ? 'chain-ok' : 'chain-fail'}`}>
          <div className="chain-icon">{chainStatus.valid ? '🔒' : '🚨'}</div>
          <div>
            <strong>{chainStatus.valid ? 'Audit Chain Intact' : 'CHAIN TAMPERED — INVESTIGATE'}</strong>
            <span className="chain-meta">
              {chainStatus.total_blocks} block{chainStatus.total_blocks !== 1 ? 's' : ''} verified
              {chainStatus.broken_at_block && ` • Broken at: ${chainStatus.broken_at_block}`}
            </span>
          </div>
          <button className="btn-sm btn-outline" onClick={async () => {
            const res = await fetch(`${API}/audit/verify`)
            setChainStatus(await res.json())
          }}>Re-verify</button>
        </div>
      )}

      {/* ── Search ─────────────────────────────── */}
      <div className="od-search">
        <input
          type="text"
          placeholder="Search by Application ID..."
          value={searchId}
          onChange={e => setSearchId(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSearch()}
        />
        <button className="btn-sm btn-blue" onClick={handleSearch}>Search</button>
      </div>
      {searchError && <div className="od-search-err">{searchError}</div>}

      <div className="od-layout">
        {/* ── Decision List ──────────────────────── */}
        <div className="od-list-panel">
          <h3 className="od-section-title">Recent Decisions</h3>
          {loading ? (
            <div className="od-loading"><span className="spinner-blue"></span> Loading...</div>
          ) : decisions.length === 0 ? (
            <div className="od-empty">No decisions recorded yet.</div>
          ) : (
            <div className="od-list">
              {decisions.map(d => (
                <div
                  key={d.block_id}
                  className={`od-item ${selected?.block_id === d.block_id ? 'od-item-active' : ''}`}
                  onClick={() => { setSelected(d); setSearchResult(null) }}
                >
                  <div className="od-item-top">
                    <span className={`od-dot ${d.decision.approved ? 'dot-green' : 'dot-red'}`}></span>
                    <span className="od-item-id">{d.application_id.slice(0, 8)}...</span>
                    <span className={`od-item-badge ${d.decision.approved ? 'ib-green' : 'ib-red'}`}>
                      {d.decision.approved ? 'You May Be Eligible' : 'You May Get Rejected'}
                    </span>
                  </div>
                  <div className="od-item-bottom">
                    <span>Prob: {(d.decision.probability * 100).toFixed(1)}%</span>
                    <span>{d.decision.confidence}</span>
                    <span className="od-item-time">{d.timestamp_iso}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ── Detail Panel ───────────────────────── */}
        <div className="od-detail-panel">
          {!detail ? (
            <div className="od-placeholder">
              <div className="od-ph-icon">🔍</div>
              <p>Select a decision from the list or search by Application ID to inspect details.</p>
            </div>
          ) : (
            <div className="od-detail" key={detail.block_id}>
              {/* Header */}
              <div className="od-detail-header">
                <div>
                  <h3>Application</h3>
                  <code className="od-app-id">{detail.application_id}</code>
                </div>
                <span className={`od-detail-badge ${detail.decision.approved ? 'ib-green' : 'ib-red'}`}>
                  {detail.decision.approved ? '✅ You May Be Eligible' : '❌ You May Get Rejected'}
                </span>
              </div>

              {/* Meta Cards */}
              <div className="od-meta-grid">
                <div className="od-meta-card">
                  <div className="omc-label">Probability</div>
                  <div className="omc-value">{(detail.decision.probability * 100).toFixed(2)}%</div>
                </div>
                <div className="od-meta-card">
                  <div className="omc-label">Confidence</div>
                  <div className={`omc-value conf-${detail.decision.confidence?.toLowerCase()}`}>
                    {detail.decision.confidence}
                  </div>
                </div>
                <div className="od-meta-card">
                  <div className="omc-label">Threshold</div>
                  <div className="omc-value">{(detail.decision.threshold * 100).toFixed(2)}%</div>
                </div>
                <div className="od-meta-card">
                  <div className="omc-label">Officer</div>
                  <div className="omc-value">{detail.officer_id}</div>
                </div>
              </div>

              {/* Anomaly flag */}
              {detail.decision.confidence === 'LOW' && (
                <div className="od-anomaly">
                  ⚠️ <strong>Anomaly Flag:</strong> This decision has low confidence and is flagged for manual review.
                </div>
              )}

              {/* SHAP Waterfall */}
              <div className="od-shap-section">
                <h4>SHAP Feature Attribution</h4>
                <div className="od-shap-base">
                  Base value (E[f(x)]): <strong>{detail.shap_output?.base_value?.toFixed(4)}</strong>
                </div>
                <div className="od-shap-bars">
                  {detail.shap_output?.top_factors?.slice(0, 10).map((f, i) => {
                    const maxAbs = Math.max(
                      ...detail.shap_output.top_factors.map(x => Math.abs(x.shap_value))
                    )
                    const pct = Math.min((Math.abs(f.shap_value) / maxAbs) * 100, 100)
                    const isPos = f.shap_value > 0
                    return (
                      <div key={i} className="od-shap-row">
                        <div className="od-shap-lbl" title={f.feature}>
                          {f.label}
                          {f.actionable && <span className="od-action-tag">actionable</span>}
                        </div>
                        <div className="od-shap-track">
                          <div
                            className={`od-shap-fill ${isPos ? 'sf-pos' : 'sf-neg'}`}
                            style={{ width: `${pct}%` }}
                          ></div>
                        </div>
                        <div className={`od-shap-val ${isPos ? 'sv-pos' : 'sv-neg'}`}>
                          {f.shap_value > 0 ? '+' : ''}{f.shap_value.toFixed(4)}
                        </div>
                        <div className="od-shap-raw">= {typeof f.raw_value === 'number' ? f.raw_value.toLocaleString() : f.raw_value}</div>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Key Inputs */}
              <details className="od-inputs-detail">
                <summary>View All Input Features ({Object.keys(detail.inputs || {}).length})</summary>
                <div className="od-inputs-grid">
                  {Object.entries(detail.inputs || {}).map(([k, v]) => (
                    <div key={k} className="od-input-item">
                      <span className="oi-key">{k}</span>
                      <span className="oi-val">{typeof v === 'number' ? v.toLocaleString() : String(v)}</span>
                    </div>
                  ))}
                </div>
              </details>

              {/* Audit Trail */}
              <div className="od-audit-section">
                <h4>🔗 Audit Chain Entry</h4>
                <div className="od-audit-grid">
                  <div><span className="oa-key">Block ID</span><code>{detail.block_id}</code></div>
                  <div><span className="oa-key">Block Hash</span><code>{detail.block_hash}</code></div>
                  <div><span className="oa-key">Prev Hash</span><code>{detail.prev_hash}</code></div>
                  <div><span className="oa-key">Timestamp</span><code>{detail.timestamp_iso}</code></div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
