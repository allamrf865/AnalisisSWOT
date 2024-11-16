import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from io import BytesIO
from fpdf import FPDF
import plotly.graph_objects as go
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
def analyze_text_with_confidence(text, qualities, confidence, category_weight):
    scores = {}
    explanations = {}
    for trait, description in qualities.items():
        trait_embedding = model.encode(description, convert_to_tensor=True)
        text_embedding = model.encode(text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(text_embedding, trait_embedding).item()
        weighted_score = similarity * (confidence / 10) * category_weight
        scores[trait] = weighted_score

        explanation = (
            f"Input: '{text}' aligns with '{trait}' ({description}). "
            f"Similarity: {similarity:.2f}. Confidence: {confidence:.2f}. "
            f"Weighted score: {weighted_score:.2f}."
        )
        explanations[trait] = explanation
    return scores, explanations

# Generate colorful bar charts using Matplotlib
def generate_bar_chart(scores, category):
    plt.figure(figsize=(10, 6))
    traits = list(scores.keys())
    values = list(scores.values())
    cmap = ListedColormap(plt.cm.tab20.colors[:len(traits)])
    colors = cmap(np.arange(len(traits)))

    plt.barh(traits, values, color=colors)
    plt.xlabel("Scores")
    plt.ylabel("Traits")
    plt.title(f"{category} Analysis", fontsize=14)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()
    return buffer

# Generate PDF Report
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

    def add_image(self, img_buffer):
        self.image(img_buffer, x=10, y=self.get_y(), w=190)
        self.ln(65)

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

# Behavioral Analysis
st.header("Behavioral Analysis")
behavioral_responses = {q: st.text_area(q) for q in BEHAVIORAL_QUESTIONS}

# SWOT Inputs
st.header("SWOT Analysis")
swot_inputs = {}
for category in ["Strengths", "Weaknesses", "Opportunities", "Threats"]:
    st.subheader(f"{category}")
    entries = []
    for i in range(3):
        text = st.text_area(f"{category} Aspect #{i+1}")
        confidence = st.slider(f"Confidence for {category} Aspect #{i+1}", 1, 10, 5)
        if text.strip():
            entries.append((text, confidence))
    swot_inputs[category] = entries

if st.button("Analyze"):
    swot_scores = {}
    swot_explanations = {}
    for category, inputs in swot_inputs.items():
        category_scores = {}
        category_explanations = {}
        for text, confidence in inputs:
            scores, explanations = analyze_text_with_confidence(
                text, LEADERSHIP_QUALITIES, confidence, CATEGORY_WEIGHTS[category]
            )
            category_scores[text] = scores
            category_explanations[text] = explanations
        swot_scores[category] = category_scores
        swot_explanations[category] = category_explanations

    # Display Results and Visualizations
    st.subheader("SWOT Analysis Results")
    for category, traits in swot_scores.items():
        st.subheader(f"{category} Breakdown")
        for text, scores in traits.items():
            st.write(f"Input: {text}")
            for trait, score in scores.items():
                st.write(f"{trait}: {score:.2f}")

            # Generate and display bar chart
            bar_chart = generate_bar_chart(scores, category)
            st.image(bar_chart, caption=f"{category} Analysis", use_column_width=True)

    # Generate PDF
    pdf_path = generate_pdf(10, swot_scores, behavioral_responses, swot_explanations)
    st.download_button("Download Full Report", open(pdf_path, "rb"), file_name="Leadership_Report.pdf", mime="application/pdf")
