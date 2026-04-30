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

export default function ConsumerLoanPortal() {
  const [customerName, setCustomerName] = useState('')
  const [productPrice, setProductPrice] = useState(50000)
  const [downPayment, setDownPayment] = useState(10000)
  const [loanAmount, setLoanAmount] = useState(40000)
  const [tenure, setTenure] = useState(12)
  const [income, setIncome] = useState(30000)
  const [dependents, setDependents] = useState(0)

  const [bankName, setBankName] = useState('HDFC')
  const [accountNumber, setAccountNumber] = useState('123456789')
  const [ifsc, setIfsc] = useState('HDFC0001')
  const [cibilScore, setCibilScore] = useState(750)
  const [existingEmi, setExistingEmi] = useState(0)
  const [creditCard, setCreditCard] = useState('yes')
  const [employmentType, setEmploymentType] = useState('salaried')

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
      formData.append('customer_name', customerName)
      formData.append('product_price', productPrice)
      formData.append('down_payment', downPayment)
      formData.append('loan_amount', loanAmount)
      formData.append('tenure', tenure)
      formData.append('income', income)
      formData.append('dependents', dependents)
      formData.append('bank_name', bankName)
      formData.append('account_number', accountNumber)
      formData.append('ifsc', ifsc)
      formData.append('cibil_score', cibilScore)
      formData.append('existing_emi', existingEmi)
      formData.append('credit_card', creditCard)
      formData.append('employment_type', employmentType)

      const res = await fetch(`${API}/consumer/evaluate`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) throw new Error(await readError(res, 'Consumer loan evaluation failed'))
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
        <h2 className="ap-hero-title">Consumer Loan Application</h2>
        <p className="ap-hero-desc">
          Share your consumer loan details and a recent bank statement. Our system will analyze
          your transactions and generate a decision instantly.
        </p>
      </div>

      <div className="ap-layout">
        <form className="ap-form" onSubmit={handleSubmit}>
          <fieldset className="ap-fieldset">
            <legend className="ap-legend"><span className="ap-legend-icon">👤</span>Applicant Details</legend>
            <div className="ap-grid">
              <div className="ap-field">
                <label>Customer Name</label>
                <input
                  type="text"
                  value={customerName}
                  onChange={(e) => setCustomerName(e.target.value)}
                  placeholder="e.g. John Doe"
                />
              </div>
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
                <label>Dependents</label>
                <input
                  type="number"
                  min={0}
                  value={dependents}
                  onChange={(e) => setDependents(e.target.value)}
                  required
                />
              </div>
            </div>
          </fieldset>

          <fieldset className="ap-fieldset">
            <legend className="ap-legend"><span className="ap-legend-icon">🧾</span>Loan Details</legend>
            <div className="ap-grid">
              <div className="ap-field">
                <label>Product Price (₹)</label>
                <input
                  type="number"
                  min={0}
                  value={productPrice}
                  onChange={(e) => setProductPrice(e.target.value)}
                  required
                />
              </div>
              <div className="ap-field">
                <label>Down Payment (₹)</label>
                <input
                  type="number"
                  min={0}
                  value={downPayment}
                  onChange={(e) => setDownPayment(e.target.value)}
                  required
                />
              </div>
              <div className="ap-field">
                <label>Loan Amount (₹)</label>
                <input
                  type="number"
                  min={0}
                  value={loanAmount}
                  onChange={(e) => setLoanAmount(e.target.value)}
                  required
                />
              </div>
              <div className="ap-field">
                <label>Tenure (Months)</label>
                <input
                  type="number"
                  min={1}
                  value={tenure}
                  onChange={(e) => setTenure(e.target.value)}
                  required
                />
              </div>
            </div>
          </fieldset>

          <fieldset className="ap-fieldset">
            <legend className="ap-legend"><span className="ap-legend-icon">🏦</span>Banking Details</legend>
            <div className="ap-grid">
              <div className="ap-field">
                <label>Bank Name</label>
                <input
                  type="text"
                  value={bankName}
                  onChange={(e) => setBankName(e.target.value)}
                  required
                />
              </div>
              <div className="ap-field">
                <label>Account Number</label>
                <input
                  type="text"
                  value={accountNumber}
                  onChange={(e) => setAccountNumber(e.target.value)}
                  required
                />
              </div>
              <div className="ap-field">
                <label>IFSC Code</label>
                <input
                  type="text"
                  value={ifsc}
                  onChange={(e) => setIfsc(e.target.value)}
                  required
                />
              </div>
              <div className="ap-field">
                <label>Existing EMI (₹)</label>
                <input
                  type="number"
                  min={0}
                  value={existingEmi}
                  onChange={(e) => setExistingEmi(e.target.value)}
                  required
                />
              </div>
              <div className="ap-field">
                <label>Has Credit Card?</label>
                <select value={creditCard} onChange={(e) => setCreditCard(e.target.value)}>
                  <option value="yes">Yes</option>
                  <option value="no">No</option>
                </select>
              </div>
              <div className="ap-field">
                <label>Employment Type</label>
                <select value={employmentType} onChange={(e) => setEmploymentType(e.target.value)}>
                  <option value="salaried">Salaried</option>
                  <option value="self-employed">Self-employed</option>
                </select>
              </div>
            </div>
          </fieldset>

          <div className="ap-field-row" style={{ marginTop: '1.5rem' }}>
            <label>Upload Bank Statement</label>
            <UploadBox
              id="consumer-statement"
              label="Bank Statement (PDF, Image, CSV)"
              icon="📄"
              accept="application/pdf,image/*,.csv"
              file={statementFile}
              onChange={setStatementFile}
            />
          </div>

          <div className="ap-actions" style={{ marginTop: '2rem' }}>
            <button type="submit" className="btn btn-primary" disabled={evaluating}>
              {evaluating ? 'Evaluating...' : '🚀 Evaluate Consumer Loan'}
            </button>
          </div>
        </form>

        <div className="ap-result-panel">
          {error && <div className="ap-card ap-error">⚠️ {error}</div>}

          {result && (
            <div className={`ap-card ap-result ${result.approved ? 'ap-approved' : 'ap-rejected'}`}>
              <div className="ap-badge-row">
                <span className={`ap-badge ${result.approved ? 'badge-green' : 'badge-red'}`}>
                  {result.approved ? '✅ You May Be Eligible' : '❌ You May Get Rejected'}
                </span>
                <span className="ap-conf">{result.decision}</span>
              </div>

              <div className="ap-summary" style={{ marginTop: '1rem' }}>
                <p><strong>Final Score:</strong> {Number(result.score).toFixed(1)} / 100</p>
                <p><strong>Document Confidence:</strong> {Number(result.txn_confidence_score).toFixed(1)} / 100</p>
                <p><strong>CIBIL Confidence:</strong> {Number(result.cibil_normalized_score).toFixed(1)} / 100</p>
                {result.statement_ml_probability !== null && result.statement_ml_probability !== undefined && (
                  <p><strong>Statement Model Probability:</strong> {(Number(result.statement_ml_probability) * 100).toFixed(1)}%</p>
                )}
                {result.reason && <p><strong>Reason:</strong> {result.reason}</p>}
              </div>

              <ExplainabilityPanel
                shapFactors={result.shap_factors}
                probability={result.score / 100}
                threshold={result.threshold_used}
                approved={result.approved}
                confidenceLevel={result.approved ? 'HIGH' : 'LOW'}
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
