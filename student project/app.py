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
<p style='text-align:center;color:gray;'>
Machine Learning Prediction with Explainable Insights
</p>
""", unsafe_allow_html=True)
st.divider()
try:
    student_data = pd.read_csv("student_ml_dataset.csv")
except:
    st.error("Dataset not found.")
    st.stop()
X = student_data[["Attendance", "StudyHours", "InternalMarks"]]
y = student_data["FinalMarks"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
model_accuracy = r2_score(y_test, predictions)
m1, m2, m3 = st.columns(3)
m1.metric("Total Students", len(student_data))
m2.metric("Model Accuracy", f"{model_accuracy:.2f}")
m3.metric("Algorithm Used", "Linear Regression")
st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Student Details")

    attendance = st.slider("Attendance (%)", 0, 100, 75)
    study_hours = st.slider("Study Hours per Day", 0.0, 6.0, 2.0)
    internal_marks = st.slider("Internal Marks", 0, 100, 70)

    predict_button = st.button("Predict Final Marks", use_container_width=True)
with right:
    st.subheader("Prediction Result")

    if predict_button:

        input_data = [[attendance, study_hours, internal_marks]]
        final_prediction = model.predict(input_data)[0]

        st.success(f"Predicted Final Marks: **{int(final_prediction)}**")

        st.divider()
        st.markdown("### AI Insights")

        insights = []

        if attendance < 70:
            insights.append("Attendance is low. Try attending classes regularly.")

        if study_hours < 2:
            insights.append("Daily study hours are less. Increase study time.")

        if internal_marks < 60:
            insights.append("Internal marks are weak. Revise fundamentals.")

        if attendance >= 85:
            insights.append("Good attendance supports learning.")

        if study_hours >= 4:
            insights.append("Strong study habit detected.")

        if internal_marks >= 80:
            insights.append("Internal marks show good understanding.")

        if len(insights) == 0:
            insights.append("Average performance. Small improvements can help.")

        for tip in insights:
            st.write("•", tip)

        st.info("Recommendation: Focus on weakest area first for better improvement.")


st.divider()
st.caption("Built by Tarun | BCA Data Science | ML Student Project")




