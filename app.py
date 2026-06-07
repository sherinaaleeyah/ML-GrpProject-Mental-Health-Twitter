import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string

# ── NLP libraries ────────────────────────────────────────────────────────────
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Risk Predictor",
    page_icon="🧠",
    layout="centered"
)

# ── Load saved artefacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model   = joblib.load("mental_health_model.pkl")
    scaler  = joblib.load("scaler.pkl")
    imputer = joblib.load("imputer.pkl")
    columns = joblib.load("feature_columns.pkl")
    return model, scaler, imputer, columns

try:
    model, scaler, imputer, FEATURE_COLS = load_artefacts()
    artefacts_loaded = True
except FileNotFoundError:
    artefacts_loaded = False

# ── Feature extraction (same pipeline as training) ───────────────────────────
analyzer   = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

FIRST_PERSON = {'i', 'me', 'my', 'myself', 'mine', "i'm", "i've", "i'd", "i'll"}
NEGATIONS    = {
    'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere',
    "can't", "won't", "don't", "doesn't", "didn't", "isn't",
    "wasn't", "couldn't", "wouldn't"
}

def extract_nlp_features(text):
    """Extract 15 NLP features from tweet text.
    Must match the feature extraction used during training exactly.
    """
    if not isinstance(text, str) or text.strip() == '':
        return [0.0] * 15

    blob        = TextBlob(text)
    polarity    = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    vader      = analyzer.polarity_scores(text)
    v_compound = vader['compound']
    v_neg      = vader['neg']
    v_pos      = vader['pos']

    tweet_len  = len(text)
    words      = text.lower().split()
    word_count = len(words)
    clean_words = [w.strip(string.punctuation) for w in words if w.strip(string.punctuation)]
    avg_word_len = np.mean([len(w) for w in clean_words]) if clean_words else 0.0

    excl_count  = text.count('!')
    ques_count  = text.count('?')
    hash_count  = len(re.findall(r'#\w+', text))
    ment_count  = len(re.findall(r'@\w+', text))

    word_set       = set(words)
    first_person   = len(word_set & FIRST_PERSON)
    negation_count = len(word_set & NEGATIONS)

    sw_count       = sum(1 for w in words if w in stop_words)
    stopword_ratio = sw_count / word_count if word_count > 0 else 0.0

    return [
        polarity, subjectivity, v_compound, v_neg, v_pos,
        tweet_len, word_count, avg_word_len,
        excl_count, ques_count, hash_count, ment_count,
        first_person, negation_count, stopword_ratio
    ]

def build_feature_row(text, followers, friends, favourites, statuses, retweets):
    """Combine engagement + NLP features into one row, matching training format."""
    # Log-transform engagement features (same as training)
    eng = [
        np.log1p(followers),
        np.log1p(friends),
        np.log1p(favourites),
        np.log1p(statuses),
        np.log1p(retweets),
    ]
    nlp = extract_nlp_features(text)
    return eng + nlp  # 5 + 15 = 20 features

def predict(feature_row):
    """Run imputation → scaling → model prediction."""
    X = pd.DataFrame([feature_row], columns=FEATURE_COLS)
    # Force all columns to float64 to match training dtype
    X = X.astype(np.float64)
    X_imp = pd.DataFrame(imputer.transform(X), columns=FEATURE_COLS).astype(np.float64)
    X_sca = pd.DataFrame(scaler.transform(X_imp), columns=FEATURE_COLS).astype(np.float64)
    prob  = model.predict_proba(X_sca)[0][1]
    label = int(prob >= 0.5)
    return label, prob

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🧠 Mental Health Risk Predictor")
st.markdown(
    "This app uses a **SMOTE + CatBoost** model trained on 20,000 tweets to predict "
    "mental health risk from social media engagement patterns."
)
st.warning(
    "⚠️ **Disclaimer:** This is a research tool only. It does not provide a "
    "clinical diagnosis. If you or someone you know needs support, please "
    "contact a mental health professional."
)

if not artefacts_loaded:
    st.error(
        "Model files not found. Make sure these four files are in the same "
        "folder as app.py:\n"
        "- `mental_health_model.pkl`\n"
        "- `scaler.pkl`\n"
        "- `imputer.pkl`\n"
        "- `feature_columns.pkl`"
    )
    st.stop()

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Single Tweet", "Batch Upload (CSV)"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Single tweet prediction
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Predict from a single tweet")
    st.markdown("Enter the tweet text and the user's engagement stats at the time of posting.")

    tweet_text = st.text_area(
        "Tweet text",
        placeholder="e.g. I can't sleep again. My brain just won't stop...",
        height=120
    )

    st.markdown("**Engagement metrics** (enter 0 if unknown)")
    col1, col2, col3 = st.columns(3)
    with col1:
        followers  = st.number_input("Followers",  min_value=0, value=0, step=1)
        friends    = st.number_input("Friends",    min_value=0, value=0, step=1)
    with col2:
        favourites = st.number_input("Favourites", min_value=0, value=0, step=1)
        statuses   = st.number_input("Statuses",   min_value=0, value=0, step=1)
    with col3:
        retweets   = st.number_input("Retweets",   min_value=0, value=0, step=1)

    if st.button("Predict", type="primary", use_container_width=True):
        if not tweet_text.strip():
            st.error("Please enter a tweet first.")
        else:
            with st.spinner("Analysing..."):
                row         = build_feature_row(tweet_text, followers, friends,
                                                favourites, statuses, retweets)
                label, prob = predict(row)

            st.divider()

            if label == 1:
                st.error(f"### 🔴 At Risk\nConfidence: **{prob*100:.1f}%**")
                st.markdown(
                    "The model detected signs associated with mental health risk in this tweet. "
                    "This is based on linguistic patterns — not a diagnosis."
                )
            else:
                st.success(f"### 🟢 Not At Risk\nConfidence: **{(1-prob)*100:.1f}%**")
                st.markdown(
                    "The model did not detect significant signs of mental health risk in this tweet."
                )

            # Show extracted features in an expander
            with st.expander("See extracted features"):
                nlp_names = [
                    'sentiment_polarity', 'sentiment_subjectivity',
                    'vader_compound', 'vader_neg', 'vader_pos',
                    'tweet_length', 'word_count', 'avg_word_length',
                    'exclamation_count', 'question_count',
                    'hashtag_count', 'mention_count',
                    'first_person_count', 'negation_count', 'stopword_ratio'
                ]
                eng_names = ['followers (log)', 'friends (log)', 'favourites (log)',
                             'statuses (log)', 'retweets (log)']
                feat_df = pd.DataFrame({
                    'Feature': eng_names + nlp_names,
                    'Value':   [round(v, 4) for v in row]
                })
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Batch CSV upload
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Predict from a CSV file")
    st.markdown(
        "Upload a CSV with these columns (same format as the training data): "
        "`post_text`, `followers`, `friends`, `favourites`, `statuses`, `retweets`"
    )

    # Download template
    template_df = pd.DataFrame({
        'post_text':  ["I can't sleep again. Feeling so empty.", "Had a great day with friends!"],
        'followers':  [200, 500],
        'friends':    [150, 300],
        'favourites': [1200, 3000],
        'statuses':   [800, 500],
        'retweets':   [0, 2],
    })
    st.download_button(
        "Download CSV template",
        data=template_df.to_csv(index=False),
        file_name="template.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_input = pd.read_csv(uploaded)
            required_cols = ['post_text', 'followers', 'friends',
                             'favourites', 'statuses', 'retweets']
            missing = [c for c in required_cols if c not in df_input.columns]

            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                st.success(f"Loaded {len(df_input)} rows. Running predictions...")

                with st.spinner("Processing all tweets..."):
                    rows = []
                    for _, row in df_input.iterrows():
                        rows.append(build_feature_row(
                            str(row['post_text']),
                            int(row.get('followers',  0)),
                            int(row.get('friends',    0)),
                            int(row.get('favourites', 0)),
                            int(row.get('statuses',   0)),
                            int(row.get('retweets',   0)),
                        ))

                    X_batch = pd.DataFrame(rows, columns=FEATURE_COLS).astype(np.float64)
                    X_batch = pd.DataFrame(imputer.transform(X_batch), columns=FEATURE_COLS).astype(np.float64)
                    X_batch = pd.DataFrame(scaler.transform(X_batch),  columns=FEATURE_COLS).astype(np.float64)

                    probs  = model.predict_proba(X_batch)[:, 1]
                    labels = (probs >= 0.5).astype(int)

                df_output = df_input.copy()
                df_output['prediction']  = labels
                df_output['risk_label']  = df_output['prediction'].map(
                    {1: "At Risk", 0: "Not At Risk"}
                )
                df_output['confidence']  = (probs * 100).round(1).astype(str) + "%"

                # Summary stats
                n_risk    = int(labels.sum())
                n_no_risk = len(labels) - n_risk
                col1, col2, col3 = st.columns(3)
                col1.metric("Total tweets",   len(labels))
                col2.metric("At Risk",         n_risk,    delta=None)
                col3.metric("Not At Risk",     n_no_risk, delta=None)

                st.dataframe(
                    df_output[['post_text', 'risk_label', 'confidence']],
                    use_container_width=True, hide_index=True
                )

                st.download_button(
                    "Download results as CSV",
                    data=df_output.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv",
                    type="primary"
                )

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "BICS 4340 / CSCI 4340 Machine Learning — Group Project · Semester II 2025/2026  \n"
    "Model: SMOTE + CatBoost · Dataset: Mental Health Twitter (Kaggle)"
)
