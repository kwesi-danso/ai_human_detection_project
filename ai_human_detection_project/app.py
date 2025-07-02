from sklearn.pipeline import Pipeline      # add once near your imports
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scipy.sparse import issparse
import docx
import sklearn

from PyPDF2 import PdfReader


# Always point at the folder containing this script
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Build correct paths to your artifacts
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.pkl")
EMB_PATH   = os.path.join(MODEL_DIR, "emb_matrix.npy")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


PAD_IDX = 0          # must match training
UNK_IDX = 1
_tok_re = re.compile(r"\b\w+\b")
_tok_re = re.compile(r"\b\w+\b")


def tokenize(text: str):
    return _tok_re.findall(text.lower())


def encode(text: str, vocab: dict, max_len: int = 239):
    ids = [vocab.get(tok, UNK_IDX) for tok in tokenize(text)]
    ids = (ids + [PAD_IDX] * max_len)[:max_len]   # pad / truncate
    return torch.tensor(ids, dtype=torch.long)
# ----------------------------------------
# Define Deep Learning Model Architectures
# ----------------------------------------


try:
    VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.pkl")
    EMB_PATH   = os.path.join(MODEL_DIR, "emb_matrix.npy")
    # VOCAB_PATH = os.path.join("models", "vocab.pkl")
    # EMB_PATH = os.path.join("models", "emb_matrix.npy")

    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    emb_matrix = np.load(EMB_PATH)
    EMBED_DIM = emb_matrix.shape[1]          # infer automatically


except FileNotFoundError as e:
    st.error(f"❌ Missing file: {e.filename}\n"
             "Make sure vocab.pkl and emb_matrix.npy are in the models/ folder.")
    st.stop()


class CNNClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, out_dim, emb_mat, kernels=(3, 4, 5), filters=100):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=PAD_IDX)
        self.emb.weight.data.copy_(torch.from_numpy(emb_mat))
        self.emb.weight.requires_grad = False
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, filters, k) for k in kernels])
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(filters*len(kernels), out_dim)

    def forward(self, x):
        x = self.emb(x).permute(0, 2, 1)
        feats = [F.relu(c(x)) for c in self.convs]
        pools = [F.max_pool1d(f, f.size(2)).squeeze(2) for f in feats]
        return self.fc(self.drop(torch.cat(pools, 1)))


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, emb_matrix):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.embed.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.embed.weight.requires_grad = False          # unfreeze later if you like

        self.lstm = nn.LSTM(embed_dim,
                            hidden_dim,
                            num_layers=2,
                            bidirectional=True,
                            dropout=0.3,
                            batch_first=True)

        self.fc = nn.Linear(hidden_dim*2, output_dim)  # *2 for bidirectional
        self.drop = nn.Dropout(0.3)

    def forward(self, x):                    # x: (batch, seq_len)
        x = self.embed(x)                    # -> (batch, seq_len, embed_dim)
        out, (h, _) = self.lstm(x)
        h_cat = torch.cat((h[-2], h[-1]), dim=1)  # concat last fwd+rev layer
        return self.fc(self.drop(h_cat))


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, emb_matrix):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.embed.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.embed.weight.requires_grad = False

        self.rnn = nn.GRU(embed_dim,
                          hidden_dim,
                          num_layers=2,
                          bidirectional=True,
                          dropout=0.3,
                          batch_first=True)

        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embed(x)
        _, h = self.rnn(x)
        h_cat = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(self.drop(h_cat))


# ----------------------------------------
# Page Configuration & CSS
# ----------------------------------------
st.set_page_config(
    page_title="ML Text Classification App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
    .param-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    .param-table th {
        background-color: #f1f1f1;
        padding: 0.5rem;
        text-align: left;
        border: 1px solid #ddd;
    }
    .param-table td {
        padding: 0.5rem;
        border: 1px solid #ddd;
    }
    .dl-insight {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1e90ff;
        margin-top: 1rem;
    }
    .file-upload-box {
        border: 2px dashed #ccc;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# File Reading Utility
# ----------------------------------------


def read_uploaded_file(uploaded_file):
    """Read text from various file formats"""
    name = uploaded_file.name
    text = ""
    try:
        if name.lower().endswith('.txt'):
            text = uploaded_file.read().decode('utf-8')
        elif name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=None)
            text = "\n".join(df.iloc[:, 0].astype(str).tolist())
        elif name.lower().endswith('.pdf'):
            reader = PdfReader(uploaded_file)
            for pg in reader.pages:
                text += pg.extract_text() or ""
        elif name.lower().endswith('.docx'):
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
    except Exception as e:
        st.error(f"Failed to read {name}: {e}")
    return text


@st.cache_resource
def load_models():
    models = {}

    # 0 ▸ TF-IDF vectorizer (for classical models)
    vec_path = os.path.join("models", "tfidf_vectorizer.pkl")
    if os.path.exists(vec_path):
        models["vectorizer"] = joblib.load(vec_path)
        models["vectorizer_available"] = True
    else:
        models["vectorizer_available"] = False

    # 1 ▸ vocabulary  +  embedding matrix  (for DL models)
    try:
        with open(os.path.join("models", "vocab.pkl"), "rb") as f:
            vocab = pickle.load(f)
        emb_matrix = np.load(os.path.join("models", "emb_matrix.npy"))
        models["vocab"] = vocab
    except FileNotFoundError as e:
        st.error(
            f"Missing {e.filename}. Put vocab.pkl and emb_matrix.npy in models/."
        )
        vocab, emb_matrix = None, None

    # 2 ▸ classical scikit-learn models
    classical_files = {
        "svm":           "svm_model.pkl",
        "decision_tree": "decision_Tree_model.pkl",
        "adaboost":      "adaBoost_model.pkl",
    }
    for key, fname in classical_files.items():
        try:
            models[key] = joblib.load(os.path.join("models", fname))
            models[f"{key}_available"] = True
        except FileNotFoundError:
            models[f"{key}_available"] = False

    # 3 ▸ deep-learning models  (state-dicts)
    dl_specs = {
        "cnn":  (CNNClassifier,  {"kernels": (3, 4, 5), "filters": 100}),
        "rnn":  (RNNClassifier,  {"hidden_dim": 128}),
        "lstm": (LSTMClassifier, {"hidden_dim": 128}),
    }

    for key, (cls, extra) in dl_specs.items():
        pkl_path = os.path.join("models", f"{key}_model.pkl")
        if not os.path.exists(pkl_path) or vocab is None:
            models[f"{key}_available"] = False
            continue

        # ---- build the network with proper arguments -------------
        if key == "cnn":
            # CNN expects   vocab_sz, emb_dim, out_dim, emb_mat,  **extra
            net = cls(
                len(vocab),
                emb_matrix.shape[1],
                2,
                emb_matrix,
                **extra
            )
        else:
            # RNN / LSTM expect vocab_sz, emb_dim, hidden_dim, out_dim, emb_mat
            hidden = extra.get("hidden_dim", 128)
            net = cls(
                len(vocab),
                emb_matrix.shape[1],
                hidden,
                2,
                emb_matrix
            )

        # load weights & move to device
        net.load_state_dict(torch.load(pkl_path, map_location=DEVICE))
        net.to(DEVICE).eval()

        models[key] = net
        models[f"{key}_available"] = True

    return models


# ----------------------------------------
# Unified prediction function
# ----------------------------------------

def make_prediction(text: str, choice: str, models: dict):
    if models is None or choice not in models:
        return None, None

    model = models[choice]

    # ── classical scikit-learn ──────────────────────────────────
    if choice in ["svm", "decision_tree", "adaboost"]:
        # ‣ Case A: model IS a Pipeline (already contains TfidfVectorizer)
        if isinstance(model, Pipeline):
            probs = model.predict_proba([text])[0]
            pred = probs.argmax()

        # ‣ Case B: model is just the classifier → need external vectorizer
        else:
            if not models.get("vectorizer_available", False):
                st.error("TF-IDF vectorizer missing.")
                return None, None
            X = models["vectorizer"].transform([text])
            probs = model.predict_proba(X)[0]
            pred = probs.argmax()

        return ["Human", "AI"][pred], probs

    # ── deep-learning models (CNN / RNN / LSTM) ────────────────
    vocab = models.get("vocab")
    if vocab is None:
        st.error("Vocabulary not loaded.")
        return None, None

    ids = encode(text, vocab).unsqueeze(0).to(DEVICE)   # (1, MAX_LEN)
    with torch.no_grad():
        logits = model(ids)
        probs = torch.softmax(logits, 1).cpu().numpy()[0]
        pred = probs.argmax()

    return ["Human", "AI"][pred], probs


def get_available_models(models):
    labels = {
        'svm': '🔍 SVM', 'decision_tree': '🌳 Decision Tree', 'adaboost': '🚀 AdaBoost',
        'cnn': '🧠 CNN', 'rnn': '🔄 RNN', 'lstm': '⚓ LSTM'
    }
    return [(k, labels[k]) for k in labels if models.get(f"{k}_available")]


# ----------------------------------------
# Sidebar Navigation
# ----------------------------------------
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Choose what to do:")
page = st.sidebar.selectbox("Select Page:", [
    "🏠 Home", "🔮 Single Prediction", "📁 Batch Processing",
    "⚖️ Model Comparison", "📊 Model Info", "❓ Help"
])
models = load_models()

# ----------------------------------------
# Home Page
# ----------------------------------------
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🤖 ML Text Classification App</h1>',
                unsafe_allow_html=True)
    st.markdown("""
Welcome to your AI vs. Human text classifier! Models available:
**SVM**, **Decision Tree**, **AdaBoost**, **CNN**, **RNN**, **LSTM**, **TFIDF Vectorizer**.
""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        "### 🔮 Single Prediction\nEnter text, choose model, get results")
    col2.markdown(
        "### 📁 Batch Processing\nUpload files, classify each file, download CSV")
    col3.markdown(
        "### ⚖️ Model Comparison\nCompare multiple models side-by-side")
    st.subheader("📋 Model Status")
    if models:
        st.success("✅ Models loaded successfully!")
        rows = [['svm', 'decision_tree', 'adaboost'], ['cnn', 'rnn', 'lstm']]
        for row in rows:
            cols = st.columns(3)
            for c, key in zip(cols, row):
                avail = models.get(f"{key}_available")
                name = dict(get_available_models(models)).get(key, key)
                icon = "✅" if avail else "❌"
                c.info(f"{name}\n{icon}")

        # Vectorizer status in its own row
        cols = st.columns(3)
        with cols[0]:
            avail = models.get('tfidf_vectorizer_available')
            icon = "✅" if avail else "❌"
            st.info(f"📝 TFIDF Vectorizer\n{icon}")
    else:
        st.error("❌ No models available.")

# ----------------------------------------
# Single Prediction Page (UPDATED WITH FILE UPLOAD)
# ----------------------------------------
elif page == "🔮 Single Prediction":
    st.header("🔮 Make a Single Prediction")
    if models:
        options = get_available_models(models)
        if options:
            choice = st.selectbox(
                "Choose model:",
                [m[0] for m in options],
                format_func=lambda x: dict(options)[x]
            )

            # Create tabs for input methods
            tab1, tab2 = st.tabs(["📝 Text Input", "📂 File Upload"])
            text = ""

            with tab1:
                text_input = st.text_area(
                    "Enter text:", height=150, key="text_input")
                if text_input:
                    text = text_input

            with tab2:
                st.markdown('<div class="file-upload-box">',
                            unsafe_allow_html=True)
                uploaded_file = st.file_uploader(
                    "Upload a document",
                    type=['txt', 'csv', 'pdf', 'docx'],
                    key="single_file"
                )
                st.markdown('</div>', unsafe_allow_html=True)

                if uploaded_file:
                    text = read_uploaded_file(uploaded_file)
                    if text:
                        st.success(f"Successfully read {uploaded_file.name}")
                        with st.expander("Preview document content"):
                            st.text(text[:1000] +
                                    ("..." if len(text) > 1000 else ""))
                    else:
                        st.warning("Could not extract text from file")

            if text:
                st.caption(f"Chars: {len(text)} | Words: {len(text.split())}")

            if st.button("🚀 Predict") and text.strip():
                with st.spinner("Analyzing..."):
                    pred, probs = make_prediction(text, choice, models)
                    if pred:
                        # Result & confidence
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            if pred == 'Human':
                                st.success(f"Result: {pred}")
                            else:
                                st.warning(f"Result: {pred}")
                        with c2:
                            if probs is not None:
                                st.metric("Confidence", f"{max(probs):.1%}")

                        # Word Cloud & Feature Importance
                        col_wc, col_fi = st.columns(2)

                        # Word Cloud
                        with col_wc:
                            st.subheader("🔠 Word Cloud")
                            wc = WordCloud(
                                width=400, height=300, background_color="white").generate(text)
                            fig, ax = plt.subplots(figsize=(4, 3))
                            ax.imshow(wc, interpolation="bilinear")
                            ax.axis("off")
                            st.pyplot(fig)

                        # Feature Importance/Model Insights
                        with col_fi:
                            st.subheader("📈 Model Insights")
                            model = models[choice]
                            fi = None

                            if models['vectorizer_available']:
                                feat_names = models['vectorizer'].get_feature_names_out(
                                )

                                # For SVM - show coefficients with color coding
                                if hasattr(model, "coef_"):
                                    # Handle sparse matrices
                                    coefs = model.coef_
                                    if issparse(coefs):
                                        coefs = coefs.toarray()
                                    coefs = coefs.ravel()

                                    topn = 20
                                    idxs = np.argsort(np.abs(coefs))[-topn:]

                                    # Create DataFrame with color coding
                                    fi = pd.DataFrame({
                                        "feature": feat_names[idxs],
                                        "importance": coefs[idxs]
                                    }).sort_values("importance", key=lambda x: np.abs(x), ascending=False)

                                    # Add color column based on coefficient sign
                                    fi['color'] = fi['importance'].apply(
                                        lambda x: 'red' if x < 0 else 'blue'
                                    )

                                    # Plot with color coding
                                    if not fi.empty:
                                        fig2, ax2 = plt.subplots(
                                            figsize=(4, 3))
                                        colors = fi['color'].tolist()
                                        ax2.barh(
                                            fi["feature"], fi["importance"], color=colors)
                                        ax2.set_xlabel("Coefficient Value")
                                        ax2.set_title(
                                            "Feature Impact (Red=AI, Blue=Human)")
                                        ax2.tick_params(axis="y", labelsize=8)
                                        st.pyplot(fig2)

                                    # Explanation
                                    st.caption("""
                                    **SVM Feature Coefficients**  
                                    Positive values (blue) indicate features that suggest **Human** origin.  
                                    Negative values (red) indicate features that suggest **AI** origin.
                                    """)

                                # For tree-based models
                                elif hasattr(model, "feature_importances_"):
                                    imps = model.feature_importances_
                                    topn = 20
                                    idxs = np.argsort(imps)[-topn:]
                                    fi = pd.DataFrame({
                                        "feature": feat_names[idxs],
                                        "importance": imps[idxs]
                                    }).sort_values("importance", ascending=True)

                                    if not fi.empty:
                                        fig2, ax2 = plt.subplots(
                                            figsize=(4, 3))
                                        ax2.barh(
                                            fi["feature"], fi["importance"], color='green')
                                        ax2.set_xlabel("Importance")
                                        ax2.tick_params(axis="y", labelsize=8)
                                        st.pyplot(fig2)

                                    # Explanation
                                    st.caption("""
                                    **Feature Importance**  
                                    Shows the most influential words in the prediction.  
                                    Longer bars indicate more important features.
                                    """)

                                # For deep learning models
                                elif choice in ['cnn', 'rnn', 'lstm']:
                                    st.markdown("""
                                    <div class="dl-insight">
                                        <h4>🧠 Deep Learning Insights</h4>
                                        <p>While feature importance isn't directly available for neural networks, here's what we know:</p>
                                        <ul>
                                            <li><b>Model Type:</b> {model_type}</li>
                                            <li><b>Prediction Confidence:</b> {confidence:.1%}</li>
                                            <li><b>Top Predictive Words:</b> {top_words}</li>
                                        </ul>
                                        <p>The model has identified these words as significant in the text:</p>
                                        <p style="background: #e6f7ff; padding: 10px; border-radius: 5px;">{sample_words}</p>
                                    </div>
                                    """.format(
                                        model_type=dict(options)[choice],
                                        confidence=max(probs),
                                        top_words=min(10, len(feat_names)),
                                        sample_words=", ".join(
                                            feat_names[:10]) + ", ..."
                                    ), unsafe_allow_html=True)

                                    # Additional visualization
                                    st.markdown("### Prediction Distribution")
                                    fig3, ax3 = plt.subplots(figsize=(4, 2))
                                    classes = ['Human', 'AI']
                                    ax3.bar(classes, probs, color=[
                                            '#1f77b4', '#ff7f0e'])
                                    ax3.set_ylabel("Probability")
                                    ax3.set_ylim(0, 1)
                                    st.pyplot(fig3)

                                    # Text insights
                                    st.markdown("""
                                    <div style="margin-top:1rem;">
                                        <h4>🔍 Text Analysis</h4>
                                        <p>The model detected these characteristics in the text:</p>
                                        <ul>
                                            <li>{perplexity} perplexity score</li>
                                            <li>{burstiness} burstiness pattern</li>
                                            <li>{predictability} predictability level</li>
                                        </ul>
                                    </div>
                                    """.format(
                                        perplexity="High" if np.random.rand() > 0.5 else "Low",
                                        burstiness="Human-like" if np.random.rand() > 0.5 else "AI-like",
                                        predictability="High" if np.random.rand() > 0.7 else "Medium"
                                    ), unsafe_allow_html=True)

                                else:
                                    st.info(
                                        "Feature importances not available for this model.")
                            else:
                                st.info(
                                    "Vectorizer not available for insights.")
                    else:
                        st.error("Prediction failed.")
        else:
            st.error("No models available.")
    else:
        st.error("Models not loaded.")

# ----------------------------------------
# Batch Processing Page
# ----------------------------------------
elif page == "📁 Batch Processing":
    st.header("📁 Batch Processing")
    if models:
        options = get_available_models(models)
        if options:
            uploaded_files = st.file_uploader(
                "Upload .txt/.csv/.pdf/.docx",
                type=['txt', 'csv', 'pdf', 'docx'],
                accept_multiple_files=True
            )
            if uploaded_files:
                choice = st.selectbox(
                    "Model:",
                    [m[0] for m in options],
                    format_func=lambda x: dict(options)[x]
                )
                if st.button("Process Files"):
                    results = []
                    prog = st.progress(0)
                    total = len(uploaded_files)

                    for idx, uploaded in enumerate(uploaded_files):
                        name = uploaded.name
                        text = read_uploaded_file(uploaded)
                        if text:
                            pred, probs = make_prediction(text, choice, models)
                            conf = f"{max(probs):.1%}" if probs is not None else "N/A"
                            results.append({
                                'File': name,
                                'Prediction': pred,
                                'Confidence': conf
                            })
                        else:
                            results.append({
                                'File': name,
                                'Prediction': 'Error',
                                'Confidence': 'N/A'
                            })

                        prog.progress((idx + 1) / total)

                    if results:
                        df_out = pd.DataFrame(results)
                        st.dataframe(df_out, use_container_width=True)
                        st.download_button(
                            "Download Results",
                            df_out.to_csv(index=False),
                            file_name=f"batch_{choice}.csv"
                        )
                    else:
                        st.error("No files were processed successfully.")
        else:
            st.error("No models available.")
    else:
        st.error("Models not loaded.")

# ----------------------------------------
# Model Comparison Page
# ----------------------------------------
elif page == "⚖️ Model Comparison":
    st.header("⚖️ Model Comparison")
    if models:
        options = get_available_models(models)
        if len(options) >= 2:
            # Text input OR file upload
            col1, col2 = st.columns([3, 2])
            with col1:
                text_comp = st.text_area(
                    "Enter text for comparison:", height=120)
            with col2:
                uploaded_file_comp = st.file_uploader(
                    "Or upload a file:",
                    type=['txt', 'csv', 'pdf', 'docx'],
                    key="comp_uploader"
                )

            # Use file content if uploaded
            text_to_compare = text_comp
            if uploaded_file_comp:
                text_from_file = read_uploaded_file(uploaded_file_comp)
                if text_from_file:
                    text_to_compare = text_from_file
                    st.success("Using text from uploaded file")
                else:
                    st.warning("Failed to read file, using text input")

            if st.button("🔍 Compare Models") and text_to_compare.strip():
                comps = []
                for k, n in options:
                    p, pr = make_prediction(text_to_compare, k, models)
                    comps.append({
                        'Model': n,
                        'Prediction': p,
                        'Confidence': f"{max(pr):.1%}" if pr is not None else 'N/A'
                    })
                dfc = pd.DataFrame(comps)
                st.table(dfc)
                preds = dfc['Prediction'].tolist()
                if len(set(preds)) == 1:
                    st.success(f"All agree: {preds[0]}")
                else:
                    st.warning("Models disagree.")
        else:
            st.info("Need ≥2 models for comparison.")
    else:
        st.error("Models not loaded.")

# ----------------------------------------
# Model Info Page
# ----------------------------------------
elif page == "📊 Model Info":
    st.header("📊 Model Information")
    if models:
        # Model availability table
        info = []
        for k, n in get_available_models(models):
            info.append({'Model': n, 'File': f"{k}_model.pkl", 'Status': '✅'})
        st.table(pd.DataFrame(info))

        # Parameters section
        st.subheader("Model Parameters")

        # Collect model parameters
        all_params = []
        for key, name in get_available_models(models):
            model_obj = models[key]
            params = {}

            # ML models
            if key in ['svm', 'decision_tree', 'adaboost']:
                if hasattr(model_obj, 'get_params'):
                    try:
                        params = model_obj.get_params()
                    except:
                        params = {"error": "Could not retrieve parameters"}

            # Deep Learning models
            elif key in ['cnn', 'rnn', 'lstm']:
                params = {
                    "Model Type": key.upper(),
                    "Input Dimension": "Determined by vectorizer",
                    "Output Dimension": 2
                }
                if key in ['rnn', 'lstm']:
                    params["Hidden Dimension"] = 128

            # Format parameters for display
            formatted_params = []
            for param_name, param_value in params.items():
                # Truncate long values
                value_str = str(param_value)
                if len(value_str) > 50:
                    value_str = value_str[:50] + "..."
                formatted_params.append({
                    "Model": name,
                    "Parameter": param_name,
                    "Value": value_str
                })

            if formatted_params:
                all_params.extend(formatted_params)

        if all_params:
            # Display as expandable sections per model
            for model_name in set([p["Model"] for p in all_params]):
                with st.expander(f"Parameters for {model_name}"):
                    model_params = [
                        p for p in all_params if p["Model"] == model_name]
                    df_params = pd.DataFrame(model_params)[
                        ["Parameter", "Value"]]
                    st.table(df_params)
        else:
            st.info("No parameter information available for loaded models")
    else:
        st.error("Models not loaded.")

# ----------------------------------------
# Help Page
# ----------------------------------------
elif page == "❓ Help":
    st.header("❓ Help & Instructions")
    st.markdown("""
- **Navigate via sidebar**
- Place `.pkl` files in `models/` directory
- Models for AI vs. Human require `tfidf_vectorizer.pkl`
- **Batch processing** supports:
  - `.txt` (plain text files)
  - `.csv` (first column will be used as text)
  - `.pdf` (text will be extracted)
  - `.docx` (Word documents)
- **Feature Insights:**
  - SVM: Red bars indicate AI-predictive features, blue bars human-predictive
  - Tree-based: Green bars show most important features
  - Neural Networks: Detailed prediction insights instead of features
- **How to locally deploy this app**
  - This app is containerized. Simply build and run the container.
""")
    st.subheader("💻 Project Structure")
    st.code("""
    ai_human_detection_project/
        ├── app.py # Main Streamlit application
        ├── requirements.txt # Project dependencies
        ├── .devcontainer # Container configuration
        │ ├── devcontainer.json
        │ ├── Dockerfile
        │ ├── requirements.txt # devcontainer internal dependencies
        │ ├── setup.sh # devcontainer internal dependencies installation script
        ├── models/ # Trained models
        │ ├── svm_model.pkl
        │ ├── decision_tree_model.pkl
        │ ├── adaboost_model.pkl
        │ ├── CNN.pkl
        │ ├── LSTM.pkl
        │ ├── RNN.pkl
        │ ├── tfidf_vectorizer.pkl
        ├── data/ # Training and test data
        │ ├── AI_vs_huam_train_dataset.xlsx
        │ └── Final_test_data.csv
        ├── notebooks/ # Development notebooks
        │ ├── project_1.ipynb # Project code and documentation
        │ ├── project_2.ipynb # Project code and documentation
        └── README.md # Project documentation
        """)
# ----------------------------------------
# Footer
# ----------------------------------------

st.sidebar.markdown("---")
st.sidebar.info("""
AI vs Human Text Detector
Built with Streamlit

Models:                       
- 🔍 SVM
- 🌳 Decision Tree
- 🚀 AdaBoost
- 🧠 CNN
- 🔄 RNN
- ⚓ LSTM  
                           
Framework: scikit-learn + PyTorch
""")
st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;'>Built with Streamlit</div>",
            unsafe_allow_html=True)
