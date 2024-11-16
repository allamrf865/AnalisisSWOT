import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go
from scipy.stats import entropy
from fpdf import FPDF
import matplotlib.pyplot as plt
import io

# Load multilingual model for NLP
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Expanded leadership qualities for analysis
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
    "Accountability": "Taking responsibility for decisions and outcomes",
    "Problem-Solving": "Analyzing and resolving challenges effectively",
    "Innovation": "Developing creative and new solutions to problems",
    "Resilience": "Recovering quickly from setbacks and staying focused",
    "Emotional Intelligence": "Managing emotions and understanding others effectively",
    "Crisis Management": "Leading effectively during high-stress situations"
}

CATEGORY_WEIGHTS = {"Strengths": 1.5, "Weaknesses": 1.2, "Opportunities": 1.4, "Threats": 1.1}
WATERMARK = "AI by Allam Rafi FKUI 2022"

# Calculate similarity scores
def calculate_scores(text, qualities, confidence, category_weight):
    scores = {}
    for trait, description in qualities.items():
        trait_embedding = model.encode(description, convert_to_tensor=True)
        text_embedding = model.encode(text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(text_embedding, trait_embedding).item()
        weighted_score = similarity * (confidence / 10) * category_weight
        scores[trait] = weighted_score
    return scores

# Entropy-based diversity scoring
def calculate_entropy(scores):
    score_array = np.array(list(scores.values()))
    if score_array.sum() == 0:
        return 0
    normalized_scores = score_array / score_array.sum()
    return entropy(normalized_scores)

# Leadership Viability Index (LSI)
def calculate_lsi(scores, behavioral_score, inconsistencies):
    epsilon = 1e-9
    positive = scores["Strengths"] + scores["Opportunities"] + behavioral_score
    negative = scores["Weaknesses"] + scores["Threats"]
    inconsistency_penalty = len(inconsistencies) * 0.1
    return np.log((positive / (negative + inconsistency_penalty + epsilon)) + epsilon)

# Generate PDF report
def generate_pdf(lsi_score, swot_breakdown, behavioral_analysis, radar_chart, scatter_plot):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'SWOT-Based Leadership Evaluation Report', ln=True, align='C')
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 10, WATERMARK, align='R', ln=True)
    pdf.ln(10)

    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "Leadership Viability Index (LSI):", ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"{lsi_score:.2f}", ln=True)
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "SWOT Breakdown:", ln=True)
    pdf.set_font('Arial', '', 12)
    for category, details in swot_breakdown.items():
        pdf.cell(0, 10, f"{category}:", ln=True)
        for text, analysis in details.items():
            pdf.cell(0, 10, f"Input: {text}", ln=True)
            for trait, score in analysis.items():
                pdf.cell(0, 10, f"  - {trait}: {score:.2f}", ln=True)
        pdf.ln(5)

    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "Behavioral Analysis:", ln=True)
    pdf.set_font('Arial', '', 12)
    for example, score in behavioral_analysis.items():
        pdf.cell(0, 10, f"Example: {example}", ln=True)
        pdf.cell(0, 10, f" - Score: {score:.2f}", ln=True)
    pdf.ln(5)

    radar_stream = io.BytesIO()
    radar_chart.savefig(radar_stream, format='png')
    radar_stream.seek(0)
    pdf.image(radar_stream, x=10, y=pdf.get_y(), w=180)
    pdf.ln(70)

    scatter_stream = io.BytesIO()
    scatter_plot.savefig(scatter_stream, format='png')
    scatter_stream.seek(0)
    pdf.image(scatter_stream, x=10, y=pdf.get_y(), w=180)
    pdf.ln(70)

    pdf_file = "/tmp/Advanced_Leadership_Report.pdf"
    pdf.output(pdf_file)
    return pdf_file

# Create visualizations
def create_visualizations(scores):
    radar_data = list(scores.values())
    qualities = list(scores.keys())
    angles = np.linspace(0, 2 * np.pi, len(qualities), endpoint=False).tolist()
    radar_data += radar_data[:1]
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.fill(angles, radar_data, color='blue', alpha=0.25)
    ax.plot(angles, radar_data, color='blue', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(qualities)
    radar_chart = plt

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(scores["Strengths"], scores["Weaknesses"], scores["Opportunities"], c=scores["Threats"], cmap='viridis')
    ax.set_xlabel('Strengths')
    ax.set_ylabel('Weaknesses')
    ax.set_zlabel('Opportunities')
    scatter_plot = plt

    return radar_chart, scatter_plot

# Streamlit App
st.title("🌟 Advanced SWOT-Based Leadership Analysis 🌟")
st.markdown("**Analyze your leadership potential with detailed input breakdown and professional recommendations.**")

# Input
swot_entries = {}
for category in ["Strengths", "Weaknesses", "Opportunities", "Threats"]:
    st.subheader(f"{category} Inputs")
    entries = []
    for i in range(3):
        text = st.text_area(f"{category} #{i + 1}", placeholder=f"Enter a {category} aspect (in English or Indonesian)...")
        confidence = st.slider(f"Confidence for {category} #{i + 1}", 1, 10, 5)
        if text:
            entries.append((text, confidence))
    swot_entries[category] = entries

# Behavioral Evidence Section
st.subheader("Behavioral Evidence")
behavioral_examples = []
for i in range(2):
    example = st.text_area(f"Behavioral Example #{i + 1}", placeholder="Describe a leadership scenario...")
    if example:
        behavioral_examples.append(example)

# Analyze Button
if st.button("Analyze"):
    swot_breakdown = {}
    scores = {}
    for category, entries in swot_entries.items():
        breakdown = {}
        for text, confidence in entries:
            analysis = calculate_scores(text, LEADERSHIP_QUALITIES, confidence, CATEGORY_WEIGHTS[category])
            breakdown[text] = analysis
        swot_breakdown[category] = breakdown
        scores[category] = sum([sum(analysis.values()) for analysis in breakdown.values()])

    behavioral_analysis = {example: len(example.split()) / 50 for example in behavioral_examples}
    lsi_score = calculate_lsi(scores, np.mean(list(behavioral_analysis.values())), [])

    radar_chart, scatter_plot = create_visualizations(scores)
    st.metric("Leadership Viability Index (LSI)", f"{lsi_score:.2f}")

    for category, breakdown in swot_breakdown.items():
        st.subheader(f"{category} Breakdown")
        for text, analysis in breakdown.items():
            st.write(f"**Input**: {text}")
            for trait, score in analysis.items():
                st.write(f"- **{trait}**: {score:.2f}")

    st.subheader("Behavioral Analysis")
    for example, score in behavioral_analysis.items():
        st.write(f"**Example**: {example}")
        st.write(f"- Score: {score:.2f}")

    pdf_file = generate_pdf(lsi_score, swot_breakdown, behavioral_analysis, radar_chart, scatter_plot)
    with open(pdf_file, "rb") as f:
        st.download_button("Download Full Report", f, file_name="Leadership_Evaluation_Report.pdf")
