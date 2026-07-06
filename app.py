import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIGURATION ----------------
st.set_page_config(page_title="Universal AI Data Scientist", layout="wide")

# ---------------- 1. THE UNIVERSAL TRAINER (Model Factory) ----------------
def train_on_any_data(df, target_col):
    """Automatically prepares and trains a model on any dataset provided."""
    data = df.copy().dropna()
    
    # Encoding: Convert all text columns (like "Airport" or "Job") to numbers
    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = le.fit_transform(data[col].astype(str))
    
    # Split features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training a new model specifically for THIS dataset
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    return model, X.columns.tolist(), accuracy

# ---------------- UI STYLING ----------------
st.markdown("""
<style>
h1 { text-align: center; color: #1f77b4; }
.stButton>button { background-color: #1f77b4; color: white; border-radius: 10px; height: 3em; width: 100%; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Universal AI Data Scientist")
st.write("Upload **any** dataset (Bank, Airlines, etc.), select a target, and the AI builds the model.")

# ---------------- 2. DYNAMIC DATA UPLOAD ----------------
st.header("1. Data & Model Training")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head(5))

    # Let the user pick what they want to predict
    target_choice = st.selectbox("Which column do you want to predict?", df.columns)

    if st.button("Build & Train This Model"):
        with st.spinner("Analyzing patterns and training model..."):
            # This handles the "Any Dataset" logic
            model, feature_names, acc = train_on_any_data(df, target_choice)
            
            # Store in session state to persist through UI interactions
            st.session_state['trained_model'] = model
            st.session_state['features'] = feature_names
            st.session_state['acc'] = acc
            st.session_state['target'] = target_choice
            
            st.success(f"Model Built Successfully! Accuracy: {acc:.2%}")

# ---------------- 3. DYNAMIC SIDEBAR ----------------
if 'trained_model' in st.session_state:
    st.sidebar.header("Current Model Stats")
    st.sidebar.write(f"Target: **{st.session_state['target']}**")
    st.sidebar.write(f"Accuracy: **{st.session_state['acc']:.2%}**")
    st.sidebar.write(f"Features Used: {len(st.session_state['features'])}")

# ---------------- 4. AI STRATEGY ASSISTANT (2026 FIX) ----------------
st.header("🤖 AI Strategy Assistant")

try:
    # Use the key from your secrets.toml
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    # FIXED: Using the 3.1-flash-lite model confirmed in your 2026 environment
    gemini_model = genai.GenerativeModel("models/gemini-3.1-flash-lite")

    user_question = st.text_input("Ask a question about your data or model results")

    if user_question:
        # Provide the AI actual context about the model just built
        current_acc = st.session_state.get('acc', 'Not trained yet')
        current_target = st.session_state.get('target', 'None')
        
        context = f"""
        You are a Senior Data Scientist. 
        The user has trained a model to predict '{current_target}'. 
        The current model accuracy is {current_acc}.
        The features used are: {st.session_state.get('features', [])}.
        """
        
        full_prompt = f"{context}\n\nUser Question: {user_question}"
        response = gemini_model.generate_content(full_prompt)
        st.info(response.text)

except Exception as e:
    st.error("AI Assistant is currently unavailable.")
    with st.expander("Technical Error Details"):
        st.write(e)
