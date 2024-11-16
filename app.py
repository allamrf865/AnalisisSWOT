import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import datetime

# Load NLP model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

LEADERSHIP_QUALITIES = {
    "Leadership": "Ability to lead and inspire others",
    "Influence": "Capability to motivate and guide people",
    "Vision": "Having a clear and inspiring direction for the future",
    "Integrity": "Consistently acting with honesty and strong ethics",
    "Confidence": "Demonstrating self-assuredness in decisions and actions",
    "Empathy": "Ability to understand and share others' feelings",
    "Team Collaboration": "Effectiveness in working with others to achieve common goals",
    "Conflict Resolution": "Managing and resolving disagreements constructively",
    "Strategic Thinking": "Ability to set long-term goals and plans",
    "Decision-Making": "Making effective choices under uncertainty or pressure",
    "Adaptability": "Flexibility to adjust to changing circumstances",
    "Time Management": "Prioritizing tasks to achieve efficiency and productivity",
    "Goal Orientation": "Focusing on achieving specific objectives",
    "Problem-Solving": "Analyzing and resolving challenges effectively",
    "Resilience": "Recovering quickly from setbacks and staying focused",
    "Crisis Management": "Leading effectively during high-stress situations"
}

CATEGORY_WEIGHTS = {"Strengths": 1.5, "Weaknesses": 1.2, "Opportunities": 1.4, "Threats": 1.1}
BEHAVIORAL_QUESTIONS = {
    "Adaptability": "How well do you adjust to new challenges or changes?",
    "Decision-Making": "How confident are you in making critical decisions?",
    "Confidence": "How self-assured are you in leadership situations?",
    "Teamwork": "How effectively do you collaborate with others?",
    "Integrity": "How consistently do you uphold strong ethical values?"
}
WATERMARK = "AI by Allam Rafi FKUI 2022"

# Dynamic NLP explanations
def analyze_text(text, qualities, confidence, category_weight):
    scores = {}
    explanations = {}
    for trait, description in qualities.items():
        trait_embedding = model.encode(description, convert_to_tensor=True)
        text_embedding = model.encode(text, convert_to_tensor=True)

        similarity = util.pytorch_cos_sim(text_embedding, trait_embedding).item()
        weighted_score = similarity * (confidence / 10) * category_weight
        scores[trait] = weighted_score

        explanation = f"Input: '{text}' aligns with '{trait}' ({description}). Similarity score: {similarity:.2f}. "
        explanation += f"Final score adjusted with confidence ({confidence}/10) and category weight ({category_weight})."
        explanations[trait] = explanation
    return scores, explanations

# Visualizations
def create_2d_chart(scores, category):
    df = pd.DataFrame(scores.items(), columns=["Trait", "Score"]).sort_values("Score", ascending=False)
    fig = px.bar(df, x="Score", y="Trait", orientation="h", color="Score", color_continuous_scale="Viridis",
                 title=f"{category} Breakdown")
    return fig

def create_3d_chart(scores):
    df = pd.DataFrame(scores.items(), columns=["Category", "Score"])
    df["Impact"] = np.random.rand(len(df))
    fig = go.Figure(data=[go.Scatter3d(
        x=df["Category"], y=df["Score"], z=df["Impact"],
        mode='markers', marker=dict(size=8, color=df["Score"], colorscale='Viridis')
    )])
    fig.update_layout(scene=dict(
        xaxis_title='Category', yaxis_title='Score', zaxis_title='Impact'
    ))
    return fig

# PDF Report
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'SWOT-Based Leadership Evaluation Report', align='C', ln=True)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f"Page {self.page_no()} - {WATERMARK}", align='C')

    def add_section(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

    def add_paragraph(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 10, text)
        self.ln(5)

def generate_pdf(lsi_score, swot_breakdown, behavioral_scores, explanations):
    pdf = PDFReport()
    pdf.add_page()

    pdf.add_section("Leadership Viability Index")
    pdf.add_paragraph(f"Your LSI Score: {lsi_score:.2f}")

    pdf.add_section("Behavioral Analysis")
    for trait, score in behavioral_scores.items():
        pdf.add_paragraph(f"{trait}: {score:.2f}")

    for category, traits in swot_breakdown.items():
        pdf.add_section(f"{category} Breakdown")
        for input_text, scores in traits.items():
            pdf.add_paragraph(f"Input: {input_text}")
            for trait, score in scores.items():
                pdf.add_paragraph(f"{trait}: {score:.2f}")
                pdf.add_paragraph(f"Explanation: {explanations[category][input_text][trait]}")

    pdf_file_path = "/tmp/Leadership_Report.pdf"
    pdf.output(pdf_file_path)
    return pdf_file_path

# Streamlit App
st.title("🌟 Advanced SWOT-Based Leadership Analysis 🌟")

# Input Sections
st.header("Behavioral Analysis")
behavioral_scores = {}
for question, description in BEHAVIORAL_QUESTIONS.items():
    score = st.slider(description, 1, 10, 5)
    behavioral_scores[question] = score

swot_inputs = {}
for category in ["Strengths", "Weaknesses", "Opportunities", "Threats"]:
    st.header(f"{category}")
    entries = []
    for i in range(3):
        text = st.text_area(f"{category} Aspect #{i+1}")
        confidence = st.slider(f"Confidence for {category} Aspect #{i+1}", 1, 10, 5)
        if text:
            entries.append((text, confidence))
    swot_inputs[category] = entries

if st.button("Analyze"):
    # Calculate scores
    swot_scores = {}
    swot_explanations = {}
    for category, inputs in swot_inputs.items():
        category_scores = {}
        category_explanations = {}
        for text, confidence in inputs:
            trait_scores, trait_explanations = analyze_text(
                text, LEADERSHIP_QUALITIES, confidence, CATEGORY_WEIGHTS[category])
            category_scores[text] = trait_scores
            category_explanations[text] = trait_explanations
        swot_scores[category] = category_scores
        swot_explanations[category] = category_explanations

    # Display results
    for category, traits in swot_scores.items():
        st.subheader(f"{category} Breakdown")
        for text, scores in traits.items():
            st.write(f"Input: {text}")
            for trait, score in scores.items():
                st.write(f"{trait}: {score:.2f}")

    # Generate PDF
    pdf_path = generate_pdf(10, swot_scores, behavioral_scores, swot_explanations)
    st.download_button("Download Full Report", open(pdf_path, "rb"), file_name="Leadership_Report.pdf", mime="application/pdf")
