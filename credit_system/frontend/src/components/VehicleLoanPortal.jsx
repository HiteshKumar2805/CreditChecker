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
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
      aria-label={`Upload ${label}`}
    >
      <input
        ref={inputRef} id={id} type="file" accept={accept} style={{ display: 'none' }}
        onChange={(e) => e.target.files[0] && onChange(e.target.files[0])}
      />
      <div className="ap-upload-icon">{file ? '✅' : icon}</div>
      {file ? (
        <div className="ap-upload-info">
          <span className="ap-file-name">{file.name}</span>
          <span className="ap-file-size">{(file.size / 1024).toFixed(1)} KB</span>
        </div>
      ) : (
        <div className="ap-upload-info">
          <span className="ap-upload-label">{label}</span>
          <span className="ap-upload-hint">Drop statement here or click</span>
        </div>
      )}

      {file && (
        <button
          className="ap-remove-btn"
          onClick={(e) => { e.stopPropagation(); onChange(null) }}
          title="Remove file"
        >
          ✕
        </button>
      )}
    </div>
  )
}

export default function VehicleLoanPortal() {
  const [cibilScore, setCibilScore] = useState(650)
  const [income, setIncome] = useState(30000)
  const [vehicleCompany, setVehicleCompany] = useState('Toyota')
  const [vehicleModel, setVehicleModel] = useState('Camry')
  const [statementFile, setStatementFile] = useState(null)

  const [evaluating, setEvaluating] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const formatShap = (value) => {
    if (value === null || value === undefined || Number.isNaN(value)) return '0.000'
    return Number(value).toFixed(3)
  }

  const readError = async (res, fallback) => {
    try {
      const data = await res.json()
      if (typeof data.detail === 'string') return data.detail
    } catch {
      // Use the generic fallback when the API did not return JSON.
    }
    return fallback
  }

  const formatSignal = (value) => {
    if (typeof value === 'number') return value.toLocaleString()
    return String(value)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!statementFile) {
      setError('Please upload a bank statement first.')
      return
    }

    setEvaluating(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('statement', statementFile)
      formData.append('cibil_score', cibilScore)
      formData.append('income', income)
      formData.append('vehicle_company', vehicleCompany)
      formData.append('vehicle_model', vehicleModel)

      const res = await fetch(`${API}/vehicle/evaluate`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) throw new Error(await readError(res, 'Vehicle loan evaluation failed'))
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setEvaluating(false)
    }
  }

  return (
    <div className="applicant-portal">
      <div className="ap-hero">
        <h2 className="ap-hero-title">Vehicle Loan Application</h2>
        <p className="ap-hero-desc">
          Upload a bank statement and provide your CIBIL score. The vehicle loan engine will
          analyze financial signals and return a confidence score.
        </p>
      </div>

      <div className="ap-layout">
        <form className="ap-form" onSubmit={handleSubmit}>
          <fieldset className="ap-fieldset">
            <legend className="ap-legend"><span className="ap-legend-icon">🚗</span>Applicant Inputs</legend>
            <div className="ap-grid">
              <div className="ap-field">
                <label>CIBIL Score</label>
                <input
                  type="number"
                  min={300}
                  max={900}
                  value={cibilScore}
                  onChange={(e) => setCibilScore(e.target.value)}
                  required
                />
              </div>
              <div className="ap-field">
                <label>Monthly Income (₹)</label>
                <input
                  type="number"
                  min={0}
                  value={income}
                  onChange={(e) => setIncome(e.target.value)}
                  required
                />
              </div>
              <div className="ap-field">
                <label>Vehicle Company</label>
                <input
                  type="text"
                  value={vehicleCompany}
                  onChange={(e) => setVehicleCompany(e.target.value)}
                  required
                />
              </div>
              <div className="ap-field">
                <label>Vehicle Model</label>
                <input
                  type="text"
                  value={vehicleModel}
                  onChange={(e) => setVehicleModel(e.target.value)}
                  required
                />
              </div>
            </div>
          </fieldset>

          <div className="ap-field-row" style={{ marginTop: '1.5rem' }}>
            <label>Upload Bank Statement</label>
            <UploadBox
              id="vehicle-statement"
              label="Bank Statement (PDF, Image)"
              icon="📄"
              accept="application/pdf,image/*"
              file={statementFile}
              onChange={setStatementFile}
            />
          </div>

          <div className="ap-actions" style={{ marginTop: '2rem' }}>
            <button type="submit" className="btn btn-primary" disabled={evaluating}>
              {evaluating ? 'Evaluating...' : '🚀 Evaluate Vehicle Loan'}
            </button>
          </div>
        </form>

        <div className="ap-result-panel">
          {error && <div className="ap-card ap-error">⚠️ {error}</div>}

          {result && (
            <div className={`ap-card ap-result ${result.approved ? 'ap-approved' : 'ap-rejected'}`}>
              <div className="ap-badge-row">
                <span className={`ap-badge ${result.approved ? 'badge-green' : 'badge-red'}`}>
                  {result.decision === 'MANUAL_REVIEW' ? '⚠️ MANUAL REVIEW' : (result.approved ? '✅ You May Be Eligible' : '❌ You May Get Rejected')}
                </span>
                <span className="ap-conf">{result.decision}</span>
              </div>

              <div className="ap-summary" style={{ marginTop: '1rem' }}>
                <p><strong>Final Score:</strong> {Number(result.final_score).toFixed(1)} / 100</p>
                <p><strong>Risk Score:</strong> {Number(result.risk_score).toFixed(1)} / 100</p>
                {result.statement_ml_probability !== null && result.statement_ml_probability !== undefined && (
                  <p><strong>Statement Model Probability:</strong> {(Number(result.statement_ml_probability) * 100).toFixed(1)}%</p>
                )}
              </div>

              {/* Removing Extracted Signals */}

              <ExplainabilityPanel
                shapFactors={result.shap_factors}
                probability={result.final_score / 100}
                threshold={0.7}
                approved={result.approved}
                confidenceLevel={result.decision === 'MANUAL_REVIEW' ? 'MEDIUM' : (result.approved ? 'HIGH' : 'LOW')}
                baseValue={result.base_value}
                counterfactuals={result.counterfactuals}
                modelVersion={result.model_version}
                blockHash={result.block_hash}
                applicantMessage={result.applicant_message}
                liteMode={true}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
