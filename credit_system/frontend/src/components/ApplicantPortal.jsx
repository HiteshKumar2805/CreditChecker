import { useState, useRef, useCallback } from 'react'
import './ApplicantPortal.css'
import ExplainabilityPanel from './ExplainabilityPanel'
import { API } from '../config.js'

function UploadBox({ id, label, icon, accept, file, onChange }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped) onChange(dropped)
  }, [onChange])

  return (
    <div
      className={`ap-upload-box ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
      onClick={() => inputRef.current?.click()}
      onDrop={handleDrop}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      role="button" tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
      aria-label={`Upload ${label}`}
    >
      <input ref={inputRef} id={id} type="file" accept={accept} style={{ display: 'none' }}
        onChange={(e) => e.target.files[0] && onChange(e.target.files[0])} />
      <div className="ap-upload-icon">{file ? '✅' : icon}</div>
      {file ? (
        <div className="ap-upload-info">
          <span className="ap-file-name">{file.name}</span>
          <span className="ap-file-size">{(file.size / 1024).toFixed(1)} KB</span>
        </div>
      ) : (
        <div className="ap-upload-info">
          <span className="ap-upload-label">{label}</span>
          <span className="ap-upload-hint">Drop image here or click</span>
        </div>
      )}
      {file && (
        <button className="ap-remove-btn" onClick={(e) => { e.stopPropagation(); onChange(null) }} title="Remove file">✕</button>
      )}
    </div>
  )
}

export default function ApplicantPortal({ kycData = null, loanType = null }) {
  const [cibilScore, setCibilScore] = useState(700)
  const [loanAmount, setLoanAmount] = useState(500000)
  const [payslipFile, setPayslipFile] = useState(null)
  const [payslipData, setPayslipData] = useState(null)
  const [extracting, setExtracting] = useState(false)
  const [evaluating, setEvaluating] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const readError = async (res, fallback) => {
    try {
      const data = await res.json()
      if (typeof data.detail === 'string') return data.detail
      if (Array.isArray(data.detail)) return data.detail.map(item => item.msg).filter(Boolean).join(', ') || fallback
    } catch { /* fallback */ }
    return fallback
  }

  const handlePayslipChange = async (file) => {
    setPayslipFile(file)
    if (!file) { setPayslipData(null); return }
    setExtracting(true); setError(null)
    try {
      const formData = new FormData()
      formData.append('payslip_img', file)
      const res = await fetch(`${API}/payslip/extract`, { method: 'POST', body: formData })
      if (!res.ok) throw new Error(await readError(res, "Failed to extract payslip"))
      setPayslipData(await res.json())
    } catch (err) { setError(err.message) }
    finally { setExtracting(false) }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!payslipData || !payslipData.salary) { setError("Please upload a valid payslip to extract salary first."); return }
    setEvaluating(true); setError(null); setResult(null)

    const company = (payslipData.company_name || "").toLowerCase()
    let tier = 3
    if (company.includes("tcs") || company.includes("infosys") || company.includes("tech")) tier = 1
    else if (company.includes("pvt") || company.includes("ltd")) tier = 2

    try {
      const res = await fetch(`${API}/evaluate_personal`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cibil_score: Number(cibilScore), monthly_salary: Number(payslipData.salary),
          company_tier: tier, loan_amount: Number(loanAmount), loan_tenure_months: 60
        }),
      })
      if (!res.ok) throw new Error(await readError(res, "Evaluation failed"))
      const data = await res.json()
      setResult(data)
      setTimeout(() => document.getElementById('ap-result-anchor')?.scrollIntoView({ behavior: 'smooth' }), 100)
    } catch (err) { setError(err.message) }
    finally { setEvaluating(false) }
  }

  return (
    <div className="applicant-portal">
      <div className="ap-hero">
        <h2 className="ap-hero-title">Personal Loan Application</h2>
        <p className="ap-hero-desc">
          Provide your CIBIL score and upload your recent salary payslip. Our system will extract the details and instantly evaluate your application using our XGBoost model.
        </p>
      </div>

      <div className="ap-layout">
        <form className="ap-form" onSubmit={handleSubmit}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
            <div className="ap-field-row">
              <label>CIBIL Score</label>
              <input type="number" value={cibilScore} onChange={e => setCibilScore(e.target.value)} min={300} max={900} required />
            </div>
            <div className="ap-field-row">
              <label>Requested Loan Amount (₹)</label>
              <input type="number" value={loanAmount} onChange={e => setLoanAmount(e.target.value)} min={50000} max={10000000} step={10000} required />
            </div>
          </div>
          <div className="ap-field-row" style={{ marginTop: '1.5rem' }}>
            <label>Upload Salary Payslip</label>
            <UploadBox id="payslip-upload" label="Payslip Image" icon="📄" accept="image/*,application/pdf" file={payslipFile} onChange={handlePayslipChange} />
          </div>
          {extracting && <p className="ap-hint">Extracting text using EasyOCR...</p>}
          {payslipData && (
            <div className="ap-extracted-data">
              <h4>Extracted Details:</h4>
              <p><strong>Company:</strong> {payslipData.company_name}</p>
              <p><strong>Salary Detected:</strong> ₹{payslipData.salary || 'Not found'}</p>
            </div>
          )}
          <div className="ap-actions" style={{ marginTop: '2rem' }}>
            <button type="submit" className="btn btn-primary" disabled={evaluating || extracting || !payslipData}>
              {evaluating ? 'Evaluating...' : '🚀 Evaluate Personal Loan'}
            </button>
          </div>
        </form>

        <div className="ap-result-panel">
          <div id="ap-result-anchor" />
          {error && <div className="ap-card ap-error">⚠️ {error}</div>}
          {result && (
            <div className={`ap-card ap-result ${result.approved ? 'ap-approved' : 'ap-rejected'}`}>
              <div className="ap-badge-row">
                <span className={`ap-badge ${result.approved ? 'badge-green' : 'badge-red'}`}>
                  {result.approved ? '✅ You May Be Eligible' : '❌ You May Get Rejected'}
                </span>
                <span className="ap-conf">{result.confidence_level} confidence</span>
              </div>
              <ExplainabilityPanel
                shapFactors={result.shap_factors} probability={result.probability}
                threshold={result.threshold_used} approved={result.approved}
                confidenceLevel={result.confidence_level} baseValue={result.base_value}
                counterfactuals={result.counterfactuals} modelVersion={result.model_version}
                applicantMessage={result.applicant_message} blockHash={result.block_hash}
                liteMode={true}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
