import streamlit as st
import pandas as pd
from google import genai
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIGURATION ----------------
st.set_page_config(page_title="Universal AI Data Scientist", layout="wide")

# ---------------- 1. MODEL TRAINING ----------------
def train_on_any_data(df, target_col):
    data = df.copy().dropna()

    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = le.fit_transform(data[col].astype(str))

    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, X.columns.tolist(), accuracy


# ---------------- UI ----------------
st.title("Universal AI Data Scientist")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Train Model"):
        model, features, acc = train_on_any_data(df, target)

        st.session_state["acc"] = acc
        st.session_state["features"] = features
        st.session_state["target"] = target

        st.success(f"Model Accuracy: {acc:.2%}")


# ---------------- SIDEBAR ----------------
if "acc" in st.session_state:
    st.sidebar.write("Accuracy:", st.session_state["acc"])
    st.sidebar.write("Target:", st.session_state["target"])


# ---------------- AI ASSISTANT (FIXED) ----------------
st.header("AI Assistant")

try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    user_question = st.text_input("Ask something about your model")

    if user_question:
        context = f"""
        Target: {st.session_state.get('target', 'N/A')}
        Accuracy: {st.session_state.get('acc', 'N/A')}
        Features: {st.session_state.get('features', [])}
        """

        prompt = context + "\n\nUser Question: " + user_question

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        st.info(response.text)

except Exception as e:
    st.error("AI Assistant error")
    st.write(e)
