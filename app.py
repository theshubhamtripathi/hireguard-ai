import streamlit as st
import joblib
from utils.preprocess import clean_text

# --- Load model ---
@st.cache_resource
def load_model():
    model = joblib.load('model/model.pkl')
    tfidf = joblib.load('model/tfidf.pkl')
    return model, tfidf

model, tfidf = load_model()

# --- Suspicious keywords ---
SUSPICIOUS_KEYWORDS = [
    'wire transfer', 'western union', 'no experience required',
    'work from home immediately', 'make money fast', 'earn up to',
    'urgently hiring', 'send money', 'bitcoin', 'gift card',
    'no interview', 'paid training', 'be your own boss',
    'guaranteed income', 'limited time offer', 'act now',
    'weekly pay', 'immediate start', 'no experience needed',
    'earn from home', 'data entry', 'multi level marketing'
]

# --- Custom CSS ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Hide default streamlit header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hero section */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99, 179, 237, 0.15);
        border: 1px solid rgba(99, 179, 237, 0.3);
        color: #63b3ed;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #63b3ed 50%, #4299e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.3rem 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #a0aec0;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* Stats bar */
    .stats-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        padding: 1.2rem 2rem;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        margin: 1.5rem 0;
    }
    .stat-item {
        text-align: center;
    }
    .stat-number {
        font-size: 1.6rem;
        font-weight: 700;
        color: #63b3ed;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Input card */
    .input-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    .input-label {
        color: #e2e8f0;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .input-hint {
        color: #718096;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    /* Text area styling */
    .stTextArea textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(99, 179, 237, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(99, 179, 237, 0.1) !important;
    }

    /* Button */
    .stButton button {
        background: linear-gradient(135deg, #3182ce 0%, #2b6cb0 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(49, 130, 206, 0.4) !important;
    }

    /* Result cards */
    .result-fake {
        background: linear-gradient(135deg, rgba(197,48,48,0.15) 0%, rgba(155,44,44,0.1) 100%);
        border: 1px solid rgba(197, 48, 48, 0.4);
        border-left: 4px solid #fc8181;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
    }
    .result-real {
        background: linear-gradient(135deg, rgba(39,103,73,0.2) 0%, rgba(22,101,52,0.1) 100%);
        border: 1px solid rgba(72, 187, 120, 0.3);
        border-left: 4px solid #68d391;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
    }
    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-desc {
        font-size: 0.9rem;
        opacity: 0.8;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #63b3ed;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    /* Keyword pills */
    .keyword-section {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .keyword-title {
        color: #e2e8f0;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .keyword-pill {
        display: inline-block;
        background: rgba(245, 101, 101, 0.15);
        border: 1px solid rgba(245, 101, 101, 0.3);
        color: #fc8181;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 4px;
    }

    /* Tips section */
    .tips-card {
        background: rgba(99, 179, 237, 0.05);
        border: 1px solid rgba(99, 179, 237, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .tip-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 6px 0;
        color: #a0aec0;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .tip-icon {
        color: #63b3ed;
        font-size: 0.9rem;
        margin-top: 2px;
        flex-shrink: 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #4a5568;
        font-size: 0.85rem;
    }
    .footer span {
        color: #e53e3e;
    }
    .footer strong {
        color: #718096;
    }

    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🛡️ AI Powered Protection</div>
    <div class="hero-title">HireGuard AI</div>
    <div class="hero-subtitle">
        Detect fake job postings instantly using Machine Learning & NLP
    </div>
</div>
""", unsafe_allow_html=True)

# --- Stats Bar ---
st.markdown("""
<div class="stats-container">
    <div class="stat-item">
        <div class="stat-number">92%</div>
        <div class="stat-label">Detection Rate</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">17K+</div>
        <div class="stat-label">Jobs Trained On</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">NLP</div>
        <div class="stat-label">Powered By</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">TF-IDF</div>
        <div class="stat-label">Algorithm</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# --- Input Section ---
st.markdown("""
<div class="input-card">
    <div class="input-label">📋 Paste Job Posting Below</div>
    <div class="input-hint">Include title, description, requirements — the more text, the better the analysis</div>
</div>
""", unsafe_allow_html=True)

job_text = st.text_area(
    label="job_input",
    label_visibility="hidden",
    height=220,
    placeholder="Example: Data Entry Specialist — Work from home, no experience needed, "
                "earn $500/day guaranteed, apply now, no interview required..."
)

analyze_clicked = st.button(
    "🔍  Analyze Job Posting",
    type="primary",
    use_container_width=True
)

# --- Analysis ---
if analyze_clicked:
    if not job_text.strip():
        st.warning("⚠️ Please paste a job posting before analyzing.")
    else:
        with st.spinner("🔄 Analyzing job posting with AI..."):
            cleaned     = clean_text(job_text)
            vectorized  = tfidf.transform([cleaned])
            prediction  = model.predict(vectorized)[0]
            proba       = model.predict_proba(vectorized)[0]
            confidence  = proba[prediction] * 100
            risk_score  = proba[1] * 100  # always fake probability

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # --- Result Banner ---
        if prediction == 1:
            st.markdown(f"""
            <div class="result-fake">
                <div class="result-title">🚨 Fake Job Detected</div>
                <div class="result-desc">
                    Our AI has flagged this job posting as potentially fraudulent.
                    Exercise extreme caution before proceeding.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-real">
                <div class="result-title">✅ Looks Like a Real Job</div>
                <div class="result-desc">
                    Our AI did not detect fraud patterns in this posting.
                    Always do your own research before applying.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # --- Metric Cards ---
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-value">{"FAKE 🚨" if prediction == 1 else "REAL ✅"}</div>
                <div class="metric-label">Verdict</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{confidence:.1f}%</div>
                <div class="metric-label">Confidence</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{risk_score:.1f}%</div>
                <div class="metric-label">Risk Score</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Confidence Progress Bar ---
        st.markdown("**📊 Risk Level:**")
        st.progress(int(risk_score))

        if risk_score < 30:
            st.caption("🟢 Low risk — looks legitimate")
        elif risk_score < 60:
            st.caption("🟡 Medium risk — proceed with caution")
        else:
            st.caption("🔴 High risk — likely fraudulent")

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # --- Suspicious Keywords ---
        found_keywords = [
            kw for kw in SUSPICIOUS_KEYWORDS
            if kw.lower() in job_text.lower()
        ]

        if found_keywords:
            pills = "".join([
                f'<span class="keyword-pill">🚩 {kw}</span>'
                for kw in found_keywords
            ])
            st.markdown(f"""
            <div class="keyword-section">
                <div class="keyword-title">⚠️ Suspicious Phrases Detected ({len(found_keywords)} found)</div>
                {pills}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="keyword-section">
                <div class="keyword-title">✅ No Suspicious Keywords Found</div>
                <div style="color: #718096; font-size: 0.9rem;">
                    No common fraud phrases were detected in this posting.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # --- Safety Tips ---
        st.markdown("""
        <div class="tips-card">
            <div class="keyword-title">💡 Stay Safe — Golden Rules</div>
            <div class="tip-item">
                <span class="tip-icon">→</span>
                <span>Never pay any money to apply for or secure a job</span>
            </div>
            <div class="tip-item">
                <span class="tip-icon">→</span>
                <span>Always research the company independently on LinkedIn or Google</span>
            </div>
            <div class="tip-item">
                <span class="tip-icon">→</span>
                <span>Be suspicious of vague job descriptions with no specific skills required</span>
            </div>
            <div class="tip-item">
                <span class="tip-icon">→</span>
                <span>Never share bank details, Aadhaar, or PAN during the application process</span>
            </div>
            <div class="tip-item">
                <span class="tip-icon">→</span>
                <span>If the salary sounds too good to be true — it probably is</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class="custom-divider"></div>
<div class="footer">
    Made with <span>❤️</span> by <strong>Shubham Tripathi</strong>
    &nbsp;·&nbsp; HireGuard AI &nbsp;·&nbsp; Built with Python, Scikit-learn & Streamlit
</div>
""", unsafe_allow_html=True)