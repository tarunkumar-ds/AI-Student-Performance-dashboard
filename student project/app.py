import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


st.set_page_config(
    page_title="AI Student Dashboard",
    page_icon="🎓",
    layout="wide"
)


st.markdown("""
<h1 style='text-align:center;'>🎓 AI Student Performance Dashboard</h1>
<p style='text-align:center; color:gray;'>
Machine Learning Prediction with Explainable AI
</p>
""", unsafe_allow_html=True)

st.divider()


df = pd.read_csv("student_ml_dataset.csv")

X = df[['Attendance', 'StudyHours', 'InternalMarks']]
y = df['FinalMarks']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = r2_score(y_test, pred)


col1, col2, col3 = st.columns(3)

col1.metric(" Students", len(df))
col2.metric(" Model Accuracy", f"{accuracy:.2f}")
col3.metric(" Algorithm", "Linear Regression")

st.divider()


left, right = st.columns(2)


with left:
    st.subheader(" Student Input")

    attendance = st.slider("Attendance (%)", 0, 100, 75)
    study = st.slider("Study Hours / Day", 0.0, 6.0, 2.0)
    internal = st.slider("Internal Marks", 0, 100, 70)

    predict_btn = st.button(" Predict Performance", use_container_width=True)


with right:
    st.subheader(" Prediction Output")

    if predict_btn:
        result = model.predict([[attendance, study, internal]])[0]

        st.success(f" Predicted Final Marks: **{int(result)}**")

        st.divider()
        st.markdown("###  AI Insight")

        insights = []

        if attendance < 70:
            insights.append(" Low attendance is a major performance risk.")

        if study < 2:
            insights.append(" Study hours are below recommended level.")

        if internal < 60:
            insights.append(" Internal assessment indicates weak fundamentals.")

        if attendance >= 85:
            insights.append(" Excellent attendance supports consistent learning.")

        if study >= 4:
            insights.append(" Strong daily study habits detected.")

        if internal >= 80:
            insights.append(" Internal marks show strong subject understanding.")

        if not insights:
            insights.append(" Average performance. Small improvements can boost results.")

        for i in insights:
            st.write(i)

        st.info(" Recommendation: Improve weakest parameters first for fastest growth.")


st.divider()
st.caption(" Built by Tarun | BCA Data Science | AI + ML Project")


