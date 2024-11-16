import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import io
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
WATERMARK = "AI by Allam Rafi FKUI 2022"

# Dynamic NLP explanations
def calculate_scores_and_explanations(text, qualities, confidence, category_weight):
    scores = {}
    explanations = {}

    for trait, description in qualities.items():
        trait_embedding = model.encode(description, convert_to_tensor=True)
        text_embedding = model.encode(text, convert_to_tensor=True)

        similarity = util.pytorch_cos_sim(text_embedding, trait_embedding).item()
        weighted_score = similarity * (confidence / 10) * category_weight
        scores[trait] = weighted_score

        # Dynamic explanation logic
        input_keywords = set(text.lower().split())
        trait_keywords = set(description.lower().split())
        overlapping_keywords = input_keywords.intersection(trait_keywords)
        missing_keywords = trait_keywords - input_keywords

        explanation = (
            f"Input: '{text}' aligns with '{trait}' ({description}). "
            f"Similarity score: {similarity:.2f}. "
        )
        if overlapping_keywords:
            explanation += f"Matched keywords: {', '.join(overlapping_keywords)}. "
        if missing_keywords:
            explanation += f"Missing elements: {', '.join(missing_keywords)}. "
        explanation += f"Final score adjusted with confidence ({confidence}/10) and category weight ({category_weight})."

        explanations[trait] = explanation

    return scores, explanations

# Create colorful 2D visualizations
def create_visualizations(scores, category):
    if not scores:
        return None
    df = pd.DataFrame(list(scores.items()), columns=["Trait", "Score"]).sort_values(by="Score", ascending=False)
    fig = px.bar(
        df, x="Score", y="Trait", orientation="h", title=f"{category} Breakdown",
        color="Score", color_continuous_scale="Viridis"
    )
    fig.update_layout(xaxis_title="Score", yaxis_title="Traits", template="plotly_dark")
    return fig

# Create 3D visualizations
def create_3d_visualization(scores):
    df = pd.DataFrame(scores.items(), columns=["Category", "Score"])
    df["Impact"] = np.random.uniform(0, 1, len(df))
    fig = go.Figure(data=[go.Scatter3d(
        x=df["Category"],
        y=df["Score"],
        z=df["Impact"],
        mode='markers',
        marker=dict(size=10, color=df["Score"], colorscale='Viridis', opacity=0.8)
    )])
    fig.update_layout(scene=dict(
        xaxis_title='Category',
        yaxis_title='Score',
        zaxis_title='Impact'
    ))
    return fig

# Leadership Viability Index (LSI)
def calculate_lsi(scores, behavioral_score):
    positive = scores.get("Strengths", 0) + scores.get("Opportunities", 0) + behavioral_score
    negative = scores.get("Weaknesses", 0) + scores.get("Threats", 0)
    return np.log((positive / (negative + 1e-9)) + 1e-9)

# Generate PDF
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'SWOT-Based Leadership Evaluation Report', align='C', ln=True)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, WATERMARK, align='R', ln=True)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def add_section(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

    def add_paragraph(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 10, text)
        self.ln(5)

    def add_signature(self):
        self.add_section("Authorized by")
        self.cell(0, 10, "Muhammad Allam Rafi, CBOA® CDSP®", ln=True)

def generate_pdf_report(lsi_score, interpretations, swot_breakdown, explanations, visualizations):
    pdf = PDFReport()
    pdf.add_page()

    pdf.add_section("Leadership Viability Index")
    pdf.add_paragraph(f"Your LSI Score: {lsi_score:.2f}")
    pdf.add_paragraph(interpretations)

    for category, breakdown in swot_breakdown.items():
        pdf.add_section(f"{category} Breakdown")
        for input_text, traits in breakdown.items():
            pdf.add_paragraph(f"Input: {input_text}")
            for trait, score in traits.items():
                pdf.add_paragraph(f"- {trait}: {score:.2f}")
                pdf.add_paragraph(f"  Explanation: {explanations[category][input_text][trait]}")

    pdf.add_signature()

    pdf_file_path = "/tmp/Leadership_Report.pdf"
    pdf.output(pdf_file_path)
    return pdf_file_path

# App execution: Ensure complete details are visible
st.button("Analyze")
