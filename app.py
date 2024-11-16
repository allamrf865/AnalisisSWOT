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
BEHAVIORAL_QUESTIONS = [
    "Describe how you handle stressful situations.",
    "Explain your approach to team leadership.",
    "What motivates you to lead others?",
    "How do you make decisions under pressure?",
    "Describe a situation where you resolved a conflict."
]
WATERMARK = "AI by Allam Rafi FKUI 2022"

# Dynamic NLP analysis
def analyze_with_confidence(text, qualities, min_conf, max_conf, category_weight):
    scores = {}
    explanations = {}
    confidence = (min_conf + max_conf) / 2  # Midpoint of confidence interval
    for trait, description in qualities.items():
        trait_embedding = model.encode(description, convert_to_tensor=True)
        text_embedding = model.encode(text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(text_embedding, trait_embedding).item()
        weighted_score = similarity * (confidence / 10) * category_weight
        scores[trait] = weighted_score

        explanation = (
            f"Input: '{text}' aligns with '{trait}' ({description}). "
            f"Similarity score: {similarity:.2f}. Confidence: {confidence:.2f}. "
            f"Weighted score: {weighted_score:.2f}."
        )
        explanations[trait] = explanation
    return scores, explanations

# Heatmap Visualization
def create_heatmap(scores):
    df = pd.DataFrame(scores).T
    fig = px.imshow(df, title="Trait Alignment Heatmap", color_continuous_scale="Viridis")
    return fig

# Radar Chart Visualization
def create_radar_chart(scores, category):
    df = pd.DataFrame(scores.items(), columns=["Trait", "Score"])
    fig = px.line_polar(df, r="Score", theta="Trait", line_close=True, title=f"{category} Radar Chart")
    fig.update_traces(fill="toself")
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

def generate_pdf(lsi_score, swot_breakdown, behavioral_responses, explanations):
    pdf = PDFReport()
    pdf.add_page()

    # Add LSI Score
    pdf.add_section("Leadership Viability Index")
    pdf.add_paragraph(f"Your LSI Score: {lsi_score:.2f}")

    # Add Behavioral Responses
    pdf.add_section("Behavioral Analysis")
    for question, response in behavioral_responses.items():
        pdf.add_paragraph(f"{question}")
        pdf.add_paragraph(f"Response: {response}")

    # Add SWOT Breakdown
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
behavioral_responses = {q: st.text_area(q) for q in BEHAVIORAL_QUESTIONS}

st.header("SWOT Analysis with Confidence Intervals")
swot_inputs = {}
for category in ["Strengths", "Weaknesses", "Opportunities", "Threats"]:
    st.subheader(f"{category}")
    entries = []
    for i in range(3):
        text = st.text_area(f"{category} Aspect #{i+1}")
        min_conf = st.slider(f"{category} Aspect #{i+1} - Min Confidence", 1, 10, 5)
        max_conf = st.slider(f"{category} Aspect #{i+1} - Max Confidence", min_conf, 10, 7)
        if text.strip():
            entries.append((text, min_conf, max_conf))
    swot_inputs[category] = entries

if st.button("Analyze"):
    # SWOT and Behavioral Analysis
    swot_scores = {}
    swot_explanations = {}
    for category, inputs in swot_inputs.items():
        category_scores = {}
        category_explanations = {}
        for text, min_conf, max_conf in inputs:
            scores, explanation = analyze_with_confidence(
                text, LEADERSHIP_QUALITIES, min_conf, max_conf, CATEGORY_WEIGHTS[category])
            category_scores[text] = scores
            category_explanations[text] = explanation
        swot_scores[category] = category_scores
        swot_explanations[category] = category_explanations

    # Display Results
    st.subheader("SWOT Results")
    for category, traits in swot_scores.items():
        st.subheader(f"{category} Breakdown")
        for text, scores in traits.items():
            st.write(f"Input: {text}")
            for trait, score in scores.items():
                st.write(f"{trait}: {score:.2f}")

    # Heatmap Visualization
    st.plotly_chart(create_heatmap(swot_scores))

    # Generate PDF
    pdf_path = generate_pdf(10, swot_scores, behavioral_responses, swot_explanations)
    st.download_button("Download Full Report", open(pdf_path, "rb"), file_name="Leadership_Report.pdf", mime="application/pdf")
