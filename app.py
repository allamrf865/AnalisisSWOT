import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF

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
WATERMARK = "AI by Allam Rafi FKUI 2022"

# Analyze SWOT inputs with confidence intervals
def analyze_with_confidence(text, qualities, min_conf, max_conf, category_weight):
    scores = {}
    explanations = {}
    confidence = (min_conf + max_conf) / 2  # Use midpoint of confidence interval
    for trait, description in qualities.items():
        trait_embedding = model.encode(description, convert_to_tensor=True)
        text_embedding = model.encode(text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(text_embedding, trait_embedding).item()
        weighted_score = similarity * (confidence / 10) * category_weight
        scores[trait] = weighted_score

        explanation = (
            f"Input: '{text}' aligns with '{trait}' ({description}). "
            f"Similarity: {similarity:.2f}, Confidence: {confidence:.2f}, "
            f"Weighted score: {weighted_score:.2f}."
        )
        explanations[trait] = explanation
    return scores, explanations

# Confidence Interval Inputs
def get_confidence_interval_input(category, i):
    min_conf = st.slider(f"{category} Aspect #{i+1} - Min Confidence", 1, 10, 5)
    max_conf = st.slider(f"{category} Aspect #{i+1} - Max Confidence", min_conf, 10, 7)
    return min_conf, max_conf

# Heatmap Visualization
def create_heatmap(scores):
    df = pd.DataFrame(scores).T
    fig = px.imshow(df, title="Trait Alignment Heatmap", color_continuous_scale="Viridis")
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

# Streamlit App
st.title("🌟 Advanced SWOT-Based Leadership Analysis 🌟")

# Input Sections
st.header("SWOT Analysis with Confidence Intervals")
swot_inputs = {}
for category in ["Strengths", "Weaknesses", "Opportunities", "Threats"]:
    st.subheader(f"{category}")
    entries = []
    for i in range(3):
        text = st.text_area(f"{category} Aspect #{i+1}")
        if text.strip():
            min_conf, max_conf = get_confidence_interval_input(category, i)
            entries.append((text, min_conf, max_conf))
    swot_inputs[category] = entries

if st.button("Analyze"):
    swot_scores = {}
    explanations = {}
    for category, inputs in swot_inputs.items():
        category_scores = {}
        category_explanations = {}
        for text, min_conf, max_conf in inputs:
            scores, explanation = analyze_with_confidence(
                text, LEADERSHIP_QUALITIES, min_conf, max_conf, CATEGORY_WEIGHTS[category])
            category_scores[text] = scores
            category_explanations[text] = explanation
        swot_scores[category] = category_scores
        explanations[category] = category_explanations

    # Heatmap
    st.subheader("Heatmap of Trait Alignments")
    st.plotly_chart(create_heatmap(swot_scores))

    # Generate PDF
    pdf = PDFReport()
    # Add sections dynamically...
    pdf_path = "/tmp/Leadership_Report.pdf"
    pdf.output(pdf_path)
    st.download_button("Download Full Report", open(pdf_path, "rb"), file_name="Leadership_Report.pdf", mime="application/pdf")
