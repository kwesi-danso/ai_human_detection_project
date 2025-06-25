# STREAMLIT ML CLASSIFICATION APP - DUAL MODEL SUPPORT
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import PyPDF2
import docx


def apply_theme(mode="Light"):
    if mode == "Light":
        st.markdown("""
        <style>
            html, body, .main {
                background-color: #ffffff;
                color: #000000;
            }
            .main-header {
                font-size: 2.8rem;
                color: #4B0082;
                text-align: center;
                margin-bottom: 2rem;
            }
            .success-box {
                padding: 1rem;
                border-radius: 0.6rem;
                background-color: #e0f7fa;
                border: 1px solid #00acc1;
                margin: 1rem 0;
            }
            .metric-card {
                background-color: #f0f4f8;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 5px solid #5e60ce;
            }
            .stButton > button {
                background-color: #5e60ce;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 5px;
            }
            .stButton > button:hover {
                background-color: #4b4fc1;
            }
        </style>
        """, unsafe_allow_html=True)

    elif mode == "Dark":
        st.markdown("""
        <style>
            html, body, .main {
                background-color: #121212;
                color: #f0f0f0;
            }
            .stApp {
                background-color: #121212;
            }
            .main-header {
                font-size: 2.8rem;
                color: #ffffff;
                text-align: center;
                margin-bottom: 2rem;
            }
            .success-box {
                padding: 1rem;
                border-radius: 0.6rem;
                background-color: #1e1e1e;
                border: 1px solid #ffffff;
                color: #ffffff;
                margin: 1rem 0;
            }
            .metric-card {
                background-color: #1f1f1f;
                color: #ffffff;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 5px solid #ffffff;
            }
            .stButton > button {
                background-color: #ffffff;
                color: #121212;
                font-weight: bold;
                border: none;
                border-radius: 5px;
            }
            .stButton > button:hover {
                background-color: #dddddd;
            }
            .stMetricLabel, .stMetricValue {
                color: #ffffff;
            }
            .stMarkdown, .stDataFrame, .stTable {
                color: #f0f0f0;
            }
        </style>
        """, unsafe_allow_html=True)


# Page Configuration
st.set_page_config(
    page_title="ML Text Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme toggle
theme_mode = st.sidebar.radio("🎨 Theme", ["Light", "Dark"])
apply_theme(theme_mode)


# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .success-box {
#         padding: 1rem;
#         border-radius: 0.5rem;
#         background-color: #d4edda;
#         border: 1px solid #c3e6cb;
#         margin: 1rem 0;
#     }
#     .metric-card {
#         background-color: #f8f9fa;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #007bff;
#     }
# </style>
# """, unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING SECTION
# ============================================================================


@st.cache_resource
def load_models():
    models = {}

    # Load SVM
    try:
        models['SVM'] = joblib.load('models/svm_model.pkl')
    except FileNotFoundError:
        st.warning("⚠️ SVM model file not found.")

    # Load Decision Tree
    try:
        models['Decision Tree'] = joblib.load('models/decision_Tree_model.pkl')
    except FileNotFoundError:
        st.warning("⚠️ Decision Tree model file not found.")

    # Load AdaBoost
    try:
        models['AdaBoost'] = joblib.load('models/adaBoost_model.pkl')
    except FileNotFoundError:
        st.warning("⚠️ AdaBoost model file not found.")

    # Load TF-IDF vectorizer (optional use)
    try:
        models['TFIDF'] = joblib.load('models/tfidf_vectorizer.pkl')
    except FileNotFoundError:
        st.warning("⚠️ TF-IDF vectorizer file not found.")

    return models


def make_prediction(text, model_choice, models):
    """Make prediction using a selected AI vs Human classification model"""
    try:
        model = models.get(model_choice)
        if model is None:
            st.warning(f"No model loaded for {model_choice}")
            return None, None

        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]

        # Assuming labels are 0 = Human, 1 = AI
        class_names = ['Human', 'AI']
        prediction_label = class_names[prediction]

        return prediction_label, probabilities

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return None, None

# 📄 File text extractors


def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    return [page.extract_text() for page in reader.pages if page.extract_text()]


def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return [para.text for para in doc.paragraphs if para.text.strip()]


def get_available_models(models):
    """Return a list of available models (value, display label) using explicit if-else logic"""
    available = []

    if models is None:
        return available

    if models.get("SVM"):
        available.append(("SVM", "🔍 SVM"))

    if models.get("Decision Tree"):
        available.append(("Decision Tree", "🌳 Decision Tree"))

    if models.get("AdaBoost"):
        available.append(("AdaBoost", "⚡ AdaBoost"))

    return available

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================


st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Choose what you want to do:")

page = st.sidebar.selectbox(
    "Select Page:",
    ["🏠 Home", "🔮 Single Prediction", "📁 Batch Processing",
        "⚖️ Model Comparison", "📊 Model Info", "❓ Help"]
)

# Load models
models = load_models()

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🤖 AI vs Human Text Classification App</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    Welcome to your machine learning web application! This app demonstrates **AI vs Human text detection**
    using multiple trained models: **Support Vector Machine (SVM)**, **Decision Tree**, and **AdaBoost**.
    """)

    # App overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🔮 Single Prediction
        - Paste text or upload a file
        - Choose a classification model
        - Get instant prediction results
        - View confidence scores
        """)

    with col2:
        st.markdown("""
        ### 📁 Batch Processing
        - Upload multiple text files
        - Analyze each for AI or Human authorship
        - Export results to CSV
        - Useful for datasets or document sets
        """)

    with col3:
        st.markdown("""
        ### ⚖️ Model Comparison
        - Compare SVM, Decision Tree, AdaBoost
        - Visualize accuracy and ROC AUC
        - Understand model strengths
        - Choose the best for deployment
        """)

    # 📋 Model Status Section
    st.subheader("📋 Model Status")

    if models:
        st.success("✅ Models loaded successfully!")

        col1, col2, col3 = st.columns(3)

        with col1:
            if models.get('SVM'):
                st.info("**🔍 SVM**\n✅ Available")
            else:
                st.warning("**🔍 SVM**\n❌ Not Available")

        with col2:
            if models.get('Decision Tree'):
                st.info("**🌳 Decision Tree**\n✅ Available")
            else:
                st.warning("**🌳 Decision Tree**\n❌ Not Available")

        with col3:
            if models.get('AdaBoost'):
                st.info("**⚡ AdaBoost**\n✅ Available")
            else:
                st.warning("**⚡ AdaBoost**\n❌ Not Available")

        # Optional: TF-IDF status if you saved it
        if models.get('TFIDF'):
            st.info("**🔤 TF-IDF Vectorizer**\n✅ Available")
        else:
            st.warning("**🔤 TF-IDF Vectorizer**\n❌ Not Available")

    else:
        st.error("❌ Models not loaded. Please check model files.")


# ============================================================================
# SINGLE PREDICTION PAGE
# ============================================================================
elif page == "🔮 Single Prediction":
    st.header("🔮 Make a Single Prediction")
    st.markdown(
        "Paste or upload text below and select a model to detect whether it was written by **AI or a human**.")

    if models:
        available_models = get_available_models(models)

        if available_models:
            # Model selection dropdown
            model_choice = st.selectbox(
                "Choose a model:",
                options=[model[0] for model in available_models],
                format_func=lambda x: next(
                    model[1] for model in available_models if model[0] == x)
            )

            # Text input
            user_input = st.text_area(
                "Paste your text here:",
                placeholder="Type or paste an essay, article, or AI-generated text...",
                height=200
            )

            # Character + word count feedback
            if user_input:
                st.caption(
                    f"📝 Character count: {len(user_input)} | Word count: {len(user_input.split())}")

            # Example prompts more suited to AI vs Human
            with st.expander("🧠 Try example texts"):
                examples = [
                    "Artificial intelligence is transforming the world across industries with unprecedented speed and scale.",
                    "Yesterday, I took a walk through the forest and was struck by how peaceful and beautiful everything was.",
                    "The utilization of transformer-based models allows for contextual embeddings and scalable inference.",
                    "I still remember my grandmother’s stories by the fire, especially the ones about her childhood in the village.",
                    "This essay explores the ethical implications of generative language models in academia and journalism."
                ]

                col1, col2 = st.columns(2)
                for i, example in enumerate(examples):
                    with col1 if i % 2 == 0 else col2:
                        if st.button(f"Example {i+1}", key=f"example_{i}"):
                            st.session_state.user_input = example
                            st.rerun()

            # Use session state for user input
            if 'user_input' in st.session_state:
                user_input = st.session_state.user_input

            # Prediction button
            if st.button("🚀 Predict", type="primary"):
                if user_input.strip():
                    with st.spinner('Analyzing text...'):
                        prediction, probabilities = make_prediction(
                            user_input, model_choice, models)

                        if prediction and probabilities is not None:
                            # Display prediction result
                            col1, col2 = st.columns([3, 1])

                            with col1:
                                if prediction == "AI":
                                    st.error(
                                        f"🤖 Prediction: **{prediction}-Generated Text**")
                                else:
                                    st.success(
                                        f"🧑 Prediction: **{prediction}-Written Text**")

                            with col2:
                                confidence = max(probabilities)
                                st.metric("Confidence", f"{confidence:.1%}")

                            # Show probability scores
                            st.subheader("📊 Prediction Probabilities")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("🧑 Human", f"{probabilities[0]:.1%}")
                            with col2:
                                st.metric("🤖 AI", f"{probabilities[1]:.1%}")

                            # Bar chart of probabilities
                            ordered_class_names = ['Human', 'AI']
                            prob_df = pd.DataFrame({
                                'Class': ordered_class_names,
                                'Probability': probabilities
                            })
                            prob_df['Class'] = pd.Categorical(
                                prob_df['Class'], categories=ordered_class_names, ordered=True)
                            st.bar_chart(prob_df.set_index(
                                'Class'), height=300)

                        else:
                            st.error("❌ Failed to make prediction.")
                else:
                    st.warning("⚠️ Please enter some text to classify.")

        else:
            st.error("No models available for prediction.")
    else:
        st.warning("Models not loaded. Please check the model files.")


# ============================================================================
# BATCH PROCESSING PAGE
# ============================================================================
elif page == "📁 Batch Processing":
    st.header("📁 Upload File for Batch Processing")
    st.markdown(
        "Upload a `.txt`,`.csv`,`.pdf`, or `.docx` file containing multiple texts to detect whether each was written by **AI or a human**.")

    if models:
        available_models = get_available_models(models)

        if available_models:
            # File upload
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['txt', 'csv', 'pdf', 'docx'],
                help="Upload a .txt, .csv, .pdf, or .docx file. For .csv, text must be in the first column."
            )

            if uploaded_file:
                # Model selection
                model_choice = st.selectbox(
                    "Choose model for batch processing:",
                    options=[model[0] for model in available_models],
                    format_func=lambda x: next(
                        model[1] for model in available_models if model[0] == x)
                )

                # Process file
                if st.button("📊 Process File"):
                    try:
                        # Read file content
                        if uploaded_file.type == "text/plain":
                            content = str(uploaded_file.read(), "utf-8")
                            texts = [line.strip()
                                     for line in content.split('\n') if line.strip()]

                        elif uploaded_file.type == "text/csv":
                            df = pd.read_csv(uploaded_file)
                            texts = df.iloc[:, 0].astype(str).tolist()

                        elif uploaded_file.type == "application/pdf":
                            texts = extract_text_from_pdf(uploaded_file)

                        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            texts = extract_text_from_docx(uploaded_file)

                        else:
                            st.error("Unsupported file type.")
                            texts = []

                        if not texts:
                            st.error("No text found in file.")
                        else:
                            st.info(f"🔍 Processing {len(texts)} texts...")

                            results = []
                            progress_bar = st.progress(0)

                            for i, text in enumerate(texts):
                                if text.strip():
                                    prediction, probabilities = make_prediction(
                                        text, model_choice, models)

                                    if prediction and probabilities is not None:
                                        results.append({
                                            'Text': text[:100] + "..." if len(text) > 100 else text,
                                            'Full_Text': text,
                                            'Prediction': prediction,
                                            'Confidence': f"{max(probabilities):.1%}",
                                            'Human_Prob': f"{probabilities[0]:.1%}",
                                            'AI_Prob': f"{probabilities[1]:.1%}"
                                        })

                                progress_bar.progress((i + 1) / len(texts))

                            if results:
                                st.success(
                                    f"✅ Successfully processed {len(results)} texts!")

                                results_df = pd.DataFrame(results)

                                # Summary statistics
                                st.subheader("📊 Summary Statistics")
                                col1, col2, col3, col4 = st.columns(4)

                                ai_count = sum(
                                    1 for r in results if r['Prediction'] == 'AI')
                                human_count = len(results) - ai_count
                                avg_confidence = np.mean(
                                    [float(r['Confidence'].strip('%')) for r in results])

                                with col1:
                                    st.metric("Total Processed", len(results))
                                with col2:
                                    st.metric("🤖 AI Predictions", ai_count)
                                with col3:
                                    st.metric(
                                        "🧑 Human Predictions", human_count)
                                with col4:
                                    st.metric("Avg Confidence",
                                              f"{avg_confidence:.1f}%")

                                # Results preview
                                st.subheader("📋 Results Preview")
                                st.dataframe(
                                    results_df[[
                                        'Text', 'Prediction', 'Confidence']],
                                    use_container_width=True
                                )

                                # Download option
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Full Results",
                                    data=csv,
                                    file_name=f"predictions_{model_choice}_{uploaded_file.name}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error("No valid texts could be processed.")

                    except Exception as e:
                        st.error(f"❌ Error processing file: {e}")
            else:
                st.info("📂 Please upload a file to get started.")

                # Show example file formats
                with st.expander("📄 Example File Formats"):
                    st.markdown("""
                    **Text File (.txt):**
                    ```
                    AI-generated content can revolutionize business workflows.
                    I wrote this essay based on my lecture notes from class.
                    Recent advances in NLP have made chatbots more human-like.
                    ```

                    **CSV File (.csv):**
                    ```
                    text
                    "AI systems are transforming education."
                    "I remember visiting the lake every summer with my family."
                    ```

                    **PDF / Word (.pdf, .docx):**
                    - Each paragraph or sentence should be written on a new line.
                    - Only the visible, readable text will be extracted (tables/images ignored).
                    - Avoid scanned or image-based PDFs — text won't be extracted.
                    """)
        else:
            st.error("❌ No models available for batch processing.")
    else:
        st.warning("⚠️ Models not loaded. Please check your model files.")

# ============================================================================
# MODEL COMPARISON PAGE
# ============================================================================
elif page == "⚖️ Model Comparison":
    st.header("⚖️ Compare Models")
    st.markdown(
        "Enter a piece of text and compare how different models classify it as AI or Human.")

    if models:
        available_models = get_available_models(models)

        if len(available_models) >= 2:
            # Text input for comparison
            comparison_text = st.text_area(
                "Enter text to compare models:",
                placeholder="Paste a paragraph to see predictions from all models...",
                height=150
            )

            if st.button("📊 Compare All Models") and comparison_text.strip():
                st.subheader("🔍 Model Comparison Results")

                comparison_results = []

                for model_key, model_name in available_models:
                    prediction, probabilities = make_prediction(
                        comparison_text, model_key, models)

                    if prediction and probabilities is not None:
                        comparison_results.append({
                            'Model': model_name,
                            'Prediction': prediction,
                            'Confidence': f"{max(probabilities):.1%}",
                            'Human %': f"{probabilities[0]:.1%}",
                            'AI %': f"{probabilities[1]:.1%}",
                            'Raw_Probs': probabilities
                        })

                if comparison_results:
                    # Display table
                    comparison_df = pd.DataFrame(comparison_results)
                    st.table(
                        comparison_df[['Model', 'Prediction', 'Confidence', 'Human %', 'AI %']])

                    # Agreement analysis
                    predictions = [r['Prediction'] for r in comparison_results]
                    if len(set(predictions)) == 1:
                        st.success(f"✅ All models agree: **{predictions[0]}**")
                    else:
                        st.warning("⚠️ Models disagree on prediction:")
                        for result in comparison_results:
                            st.write(
                                f"- {result['Model']}: **{result['Prediction']}**")

                    # Probability bar charts
                    st.subheader("📊 Detailed Probability Comparison")
                    cols = st.columns(len(comparison_results))

                    for i, result in enumerate(comparison_results):
                        with cols[i]:
                            st.write(f"**{result['Model']}**")
                            chart_data = pd.DataFrame({
                                'Class': ['Human', 'AI'],
                                'Probability': result['Raw_Probs']
                            })
                            st.bar_chart(chart_data.set_index('Class'))

                else:
                    st.error("❌ Failed to generate predictions from the models.")

        elif len(available_models) == 1:
            st.info(
                "Only one model is available. Use the Single Prediction page for analysis.")
        else:
            st.error("❌ No models available for comparison.")
    else:
        st.warning("⚠️ Models not loaded. Please check your model files.")


# ============================================================================
# MODEL INFO PAGE
# ============================================================================
elif page == "📊 Model Info":
    st.header("📊 Model Information")

    if models:
        st.success("✅ Models are loaded and ready!")

        # Model details
        st.subheader("🔧 Available Models")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            ### 🔍 Support Vector Machine (SVM)
            **Type:** Margin-based Classifier  
            **Algorithm:** SVM with probability estimation  
            **Features:** TF-IDF (up to Triagrams)  
            
            **Strengths:**
            - Strong for high-dimensional text data
            - Effective for binary classification
            - High generalization ability
            """)

        with col2:
            st.markdown("""
            ### 🌳 Decision Tree
            **Type:** Tree-based Classifier  
            **Algorithm:** Gini-based recursive splitting  
            **Features:** TF-IDF  
            
            **Strengths:**
            - Easy to interpret
            - Handles non-linear boundaries
            - Fast training
            """)

        with col3:
            st.markdown("""
            ### ⚡ AdaBoost
            **Type:** Ensemble Classifier  
            **Algorithm:** Boosted decision stumps  
            **Features:** TF-IDF  
            
            **Strengths:**
            - Boosts weak learners into strong predictions
            - Robust to overfitting
            - Good performance on small and medium datasets
            """)

        # Feature engineering info
        st.subheader("🔤 Feature Engineering")
        st.markdown("""
        **Vectorization Method:** TF-IDF (Term Frequency-Inverse Document Frequency)
        - **N-grams:** Unigrams,Bigrams and Triagrams
        - **Max Features:** Typically 1000–10000
        - **Min Document Frequency:** e.g., 2
        - **Stop Words:** Removed (English)
        """)

        # Model files status
        st.subheader("📁 Model Files Status")
        file_status = []

        files_to_check = [
            ("svm_model.pkl", "SVM Pipeline", 'SVM' in models),
            ("decision_Tree_model.pkl", "Decision Tree Pipeline",
             'Decision Tree' in models),
            ("adaBoost_model.pkl", "AdaBoost Pipeline", 'AdaBoost' in models),
            ("tfidf_vectorizer.pkl",
             "TF-IDF Vectorizer (separate file)", 'TFIDF' in models)
        ]

        for filename, description, status in files_to_check:
            file_status.append({
                "File": filename,
                "Description": description,
                "Status": "✅ Loaded" if status else "❌ Not Found"
            })

        st.table(pd.DataFrame(file_status))

        # Training info
        st.subheader("📚 Training Information")
        st.markdown("""
        **Dataset:** AI vs Human text classification dataset  
        - **Classes:** `AI` and `Human`  
        - **Preprocessing:** Removing stopwords and punctuations,Lowercasing, lemmatization, TF-IDF vectorization  
        - **Model Training:** 5-fold cross-validation, grid search for hyperparameters 
        - **Evaluation:** Evaluated on test data.   
        """)
    else:
        st.warning("⚠️ Models not loaded. Please check the `models/` directory.")

# ============================================================================
# HELP PAGE
# ============================================================================
elif page == "❓ Help":
    st.header("❓ How to Use This App")

    with st.expander("🔮 Single Prediction"):
        st.write("""
        1. **Select a model** from the dropdown (SVM, Decision Tree, or AdaBoost)
        2. **Paste or type text** into the input box (e.g., article, essay, AI-generated output)
        3. **Click 'Predict'** to get AI vs Human classification
        4. **View results:** predicted class, confidence score, and a probability breakdown
        5. **Try examples:** Click an example button to autofill test input
        """)

    with st.expander("📁 Batch Processing"):
        st.write("""
        1. **Prepare your input file:**
           - **.txt file:** One text per line
           - **.csv file:** Text should be in the **first column**
           - **.pdf/doc file:** Text should be in the **first column**
        2. **Upload your file** using the uploader
        3. **Choose a model** to run predictions on all inputs
        4. **Click 'Process File'** to classify the texts
        5. **View and download results** with predictions and confidence scores
        """)

    with st.expander("⚖️ Model Comparison"):
        st.write("""
        1. **Paste a single text** into the input field
        2. **Click 'Compare All Models'** to view predictions from all models
        3. **Check agreement:** See if the models agree on classification
        4. **Review probabilities:** Visualize each model’s confidence using bar charts
        """)

    with st.expander("🔧 Troubleshooting"):
        st.write("""
        **Common Issues and Solutions:**
        
        **Models not loading:**
        - Ensure the following `.pkl` files are in your `models/` folder:
          - `svm_model.pkl`
          - `decision_Tree_model.pkl`
          - `adaBoost_model.pkl`
          - `tfidf_vectorizer.pkl` (if used separately)
        
        **Prediction errors:**
        - Check that the input is not empty
        - Use moderately sized text (1–5 paragraphs for best results)
        - Input must be clean and in plain English
        
        **File upload issues:**
        - Only `.txt` or `.csv` files are accepted
        - Make sure the file is UTF-8 encoded
        - For `.csv` files, ensure the **first column** contains the text
        """)

    # System structure overview
    st.subheader("💻 Project Structure Overview")
    st.code("""
    ai_huam_detection_project/
    ├── app.py                  # Main Streamlit app
    ├── requirements.txt        # Python dependencies
    ├── models/                 # Trained models
    │   ├── adaBoost_model.pkl
    │   ├── decision_Tree_model.pkl
    │   ├── svm_model.pkl
    │   └── tfidf_vectorizer.pkl
    └── data/            # (Optional) test files for batch input
        ├── AI_vs_huam_dataset.xlsx
        └── Final_test_data.csv
    """)


# ============================================================================
# FOOTER
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 App Information")
st.sidebar.info("""
**AI vs Human Text Classification App**  
Built with Streamlit

**Models:**  
- 🔍 SVM  
- 🌳 Decision Tree  
- ⚡ AdaBoost  

**Framework:** scikit-learn  
**Deployment:** Streamlit Cloud Ready
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    Built with ❤️ using Streamlit | AI vs Human Text Classification Demo<br>
    <small>Developed for educational purposes in **Machine Learning** and **AI Text Detection**</small><br>
    <small>This app uses traditional ML models to detect AI-generated vs human-written text</small>
</div>
""", unsafe_allow_html=True)
