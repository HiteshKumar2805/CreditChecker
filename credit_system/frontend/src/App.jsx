import { useState } from 'react'
import KYCPortal      from './components/KYCPortal.jsx'
import ApplicantPortal from './components/ApplicantPortal.jsx'
import ConsumerLoanPortal from './components/ConsumerLoanPortal.jsx'
import VehicleLoanPortal from './components/VehicleLoanPortal.jsx'
import OfficerDashboard from './components/OfficerDashboard.jsx'
import './App.css'

export default function App() {
  const [activeTab, setActiveTab] = useState('kyc')
  const [kycResult, setKycResult] = useState(null)  // { loanType, kycData }

  // Called by KYCPortal when user picks a loan type
  const handleLoanSelected = (result) => {
    setKycResult(result)
    // Auto-navigate to the application tab
    setActiveTab('applicant')
  }

  const renderApplication = () => {
    if (kycResult?.loanType === 'consumer') {
      return <ConsumerLoanPortal kycData={kycResult?.kycData} loanType={kycResult?.loanType} />
    }
    if (kycResult?.loanType === 'vehicle') {
      return <VehicleLoanPortal kycData={kycResult?.kycData} loanType={kycResult?.loanType} />
    }
    return <ApplicantPortal kycData={kycResult?.kycData} loanType={kycResult?.loanType} />
  }

  return (
    <div className="app">
      {/* ── Header ────────────────────────────────────── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo-group">
            <div className="logo-icon">
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                <circle cx="16" cy="16" r="15" stroke="#fff" strokeWidth="2"/>
                <path d="M10 16h12M16 10v12" stroke="#fff" strokeWidth="2" strokeLinecap="round"/>
                <circle cx="16" cy="16" r="6" stroke="#f5d060" strokeWidth="1.5"/>
              </svg>
            </div>
            <div>
              <h1 className="logo-title">CreditChecker</h1>
              <p className="logo-subtitle">AI-Powered Explainable Lending</p>
            </div>
          </div>
        </div>
      </header>

      {/* ── Main Content ──────────────────────────────── */}
      <main className="main">
        {activeTab === 'kyc'       && <KYCPortal onLoanSelected={handleLoanSelected} />}
        {activeTab === 'applicant' && renderApplication()}
        {activeTab === 'officer'   && <OfficerDashboard />}
      </main>

      {/* ── Footer ────────────────────────────────────── */}
      <footer className="footer">
        <p>CreditChecker &mdash; EasyOCR KYC &bull; LightGBM + SHAP Explainability &bull; Tamper-Evident Audit Chain</p>
      </footer>
    </div>
  )
}
