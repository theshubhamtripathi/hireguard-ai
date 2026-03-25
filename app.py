import streamlit as st
import joblib
from utils.preprocess import clean_text

# --- Load model and vectorizer ---
# @st.cache_resource means: load once, reuse every time
# Without this, model reloads on every single click — very slow
@st.cache_resource
def load_model():
    model = joblib.load('model/model.pkl')
    tfidf = joblib.load('model/tfidf.pkl')
    return model, tfidf

model, tfidf = load_model()

# --- Suspicious keywords list ---
# Why: even if model says "real", these phrases are red flags
# We scan the raw text and highlight them for the user
SUSPICIOUS_KEYWORDS = [
    'wire transfer', 'western union', 'no experience required',
    'work from home immediately', 'make money fast', 'earn up to',
    'urgently hiring', 'send money', 'bitcoin', 'gift card',
    'no interview', 'paid training', 'be your own boss',
    'guaranteed income', 'limited time offer', 'act now',
    'weekly pay', 'immediate start', 'no experience needed',
    'earn from home', 'data entry', 'multi level marketing'
]

# --- Page config ---
st.set_page_config(
    page_title="HireGuard AI",
    page_icon="🛡️",
    layout="centered"
)

# --- Header ---
st.title("🛡️ HireGuard AI")
st.markdown("### Fake Job Detection System")
st.markdown("Paste any job posting below and AI will tell you if it's real or fake.")
st.divider()

# --- Input area ---
# Why text_area: job postings are long — needs a big input box
job_text = st.text_area(
    "Paste the full job posting here (title + description + requirements)",
    height=250,
    placeholder="Example: Data Entry Specialist — Work from home, "
                "no experience needed, earn $500/day, apply now..."
)

# --- Analyze button ---
if st.button("🔍 Analyze Job Posting", type="primary", use_container_width=True):

    # Handle empty input
    if not job_text.strip():
        st.warning("⚠️ Please paste a job posting first.")

    else:
        # Show a spinner while processing
        with st.spinner("Analyzing job posting..."):

            # Step 1: Clean the text same way we cleaned training data
            cleaned = clean_text(job_text)

            # Step 2: Convert to TF-IDF numbers
            vectorized = tfidf.transform([cleaned])

            # Step 3: Get prediction (0=Real, 1=Fake)
            prediction = model.predict(vectorized)[0]

            # Step 4: Get confidence score
            # predict_proba returns [prob_real, prob_fake]
            # we take the probability of whichever class was predicted
            proba = model.predict_proba(vectorized)[0]
            confidence = proba[prediction] * 100

        st.divider()

        # --- Show result ---
        if prediction == 1:
            st.error("🚨 WARNING: This looks like a FAKE job posting!")
        else:
            st.success("✅ This looks like a REAL job posting.")

        # --- Show confidence score as two metric cards ---
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Prediction",
                value="FAKE 🚨" if prediction == 1 else "REAL ✅"
            )
        with col2:
            st.metric(
                label="Confidence Score",
                value=f"{confidence:.1f}%"
            )

        # --- Show confidence bar ---
        st.markdown("**Confidence Level:**")
        st.progress(int(confidence))

        st.divider()

        # --- Suspicious keyword detection ---
        # Why: model gives one verdict but keywords show WHY it flagged it
        # This makes the app more transparent and trustworthy
        found_keywords = [
            kw for kw in SUSPICIOUS_KEYWORDS
            if kw.lower() in job_text.lower()
        ]

        if found_keywords:
            st.markdown("### 🔎 Suspicious Phrases Detected:")
            st.markdown("These phrases commonly appear in fake job postings:")
            for kw in found_keywords:
                st.markdown(f"🚩 `{kw}`")
        else:
            if prediction == 1:
                st.info("💡 No specific suspicious keywords found, "
                       "but writing style and patterns suggest this may be fake.")
            else:
                st.info("✅ No suspicious keywords detected.")

        st.divider()

        # --- Tips section ---
        st.markdown("### 💡 How to stay safe:")
        st.markdown("""
        - Never pay money to apply for a job
        - Research the company independently before applying  
        - Be suspicious of vague job descriptions
        - Never share personal banking details during job applications
        - If salary sounds too good to be true, it probably is
        """)

# --- Footer ---
st.divider()
st.caption("HireGuard AI — Built with Python, Scikit-learn & Streamlit")