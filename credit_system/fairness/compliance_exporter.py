import json
import datetime

def generate_compliance_report_html(fairness_report: dict, drift_report: dict) -> str:
    """
    Generates a full HTML compliance report conforming to regulatory standards,
    incorporating fairness metrics and model drift.
    """
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Regulatory Compliance & Fairness Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 0 auto; padding: 2rem; }}
            h1, h2, h3 {{ color: #111; border-bottom: 1px solid #eaeaea; padding-bottom: 0.5rem; }}
            .header {{ text-align: center; margin-bottom: 3rem; }}
            .summary-box {{ background: #f8f9fa; border-left: 4px solid #3b82f6; padding: 1rem; margin: 1.5rem 0; }}
            .status-pass {{ color: #059669; font-weight: bold; }}
            .status-fail {{ color: #dc2626; font-weight: bold; }}
            .status-warning {{ color: #d97706; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; margin: 1rem 0 2rem 0; font-size: 0.9rem; }}
            th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
            th {{ background-color: #f4f4f4; }}
            .metric-val {{ font-family: monospace; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>AI Credit Decisioning — Regulatory Compliance Report</h1>
            <p>Generated: {date_str} | Standard: RBI / GDPR / CFPB Alignment</p>
        </div>

        <div class="summary-box">
            <h3>Executive Summary</h3>
            <p>This report documents the disparate impact and demographic fairness of the deployed ML model, along with data drift tracking to ensure stability over time.</p>
        </div>

        <h2>1. Demographic Fairness Analysis</h2>
        <p>Fairness evaluated using the EEOC 4/5ths Rule (Disparate Impact Ratio >= 0.8) and Equalized Odds framework.</p>
    """

    for attr, metrics in fairness_report.items():
        compliance_class = "status-pass" if metrics.get("compliance_flag") == "PASS" else "status-warning"
        
        html += f"""
        <h3>Protected Attribute: {attr.replace('_', ' ').title()}</h3>
        <p>Status: <span class="{compliance_class}">{metrics.get("compliance_flag", "UNKNOWN")}</span></p>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Regulatory Target</th>
            </tr>
            <tr>
                <td>Demographic Parity Difference</td>
                <td class="metric-val">{metrics['demographic_parity_difference']}</td>
                <td>< 0.10</td>
            </tr>
            <tr>
                <td>Equalized Odds Difference</td>
                <td class="metric-val">{metrics['equalized_odds_difference']}</td>
                <td>< 0.10</td>
            </tr>
            <tr>
                <td>4/5ths Ratio (EEOC)</td>
                <td class="metric-val">{metrics['four_fifths_ratio']}</td>
                <td>>= 0.80</td>
            </tr>
            <tr>
                <td>Disadvantaged Group</td>
                <td>{metrics['disadvantaged_group']}</td>
                <td>N/A</td>
            </tr>
        </table>
        """

    html += f"""
        <h2>2. Model Stability & Drift Tracking</h2>
        <p>Evaluates if the distribution of production predictions has significantly diverged from the training baseline using the Population Stability Index (PSI).</p>
    """
    
    if drift_report:
        drift_class = "status-fail" if drift_report.get("status") == "CRITICAL" else ("status-warning" if drift_report.get("status") == "WARNING" else "status-pass")
        html += f"""
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>Population Stability Index (PSI)</td>
                <td class="metric-val">{drift_report.get('psi_score', 0)}</td>
                <td class="{drift_class}">{drift_report.get('status', 'UNKNOWN')}</td>
            </tr>
            <tr>
                <td>Baseline Mean Probability</td>
                <td class="metric-val">{drift_report.get('baseline_mean_prob', 0)}</td>
                <td>-</td>
            </tr>
            <tr>
                <td>Production Mean Probability</td>
                <td class="metric-val">{drift_report.get('recent_mean_prob', 0)}</td>
                <td>-</td>
            </tr>
        </table>
        <p><strong>Diagnosis:</strong> {drift_report.get('message', '')}</p>
        """
    else:
        html += "<p><em>Drift data currently unavailable.</em></p>"

    html += """
        <hr>
        <p style="text-align:center; font-size: 0.8rem; color: #666;">
            Confidential & Proprietary. Contains automated auditing metrics verifying compliance with Section 5 of the FTC Act / ECOA / Algorithmic Accountability acts.
        </p>
    </body>
    </html>
    """
    return html
