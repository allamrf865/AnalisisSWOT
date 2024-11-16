import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
import plotly.graph_objects as go
import plotly.express as px
import datetime

# Set Streamlit page configuration
st.set_page_config(page_title="SWOT-Based Leadership Analysis", page_icon="🌟", layout="wide")

# Load NLP Model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Define Leadership Traits
LEADERSHIP_QUALITIES_POSITIVE = {
    "Leadership": "Ability to lead and inspire others",
    "Influence": "Capability to motivate and guide people",
    "Vision": "Having a clear and inspiring direction for the future",
    "Integrity": "Consistently acting with honesty and strong ethics",
    "Confidence": "Demonstrating self-assuredness in decisions and actions",
    "Empathy": "Ability to understand and share others' feelings",
    "Team Collaboration": "Effectiveness in working with others to achieve common goals",
    "Conflict Resolution": "Managing and resolving disagreements constructively",
    "Strategic Thinking": "Ability to set long-term goals and plans"
}

LEADERSHIP_QUALITIES_NEUTRAL = {
    "Adaptability": "Flexibility to adjust to changing circumstances",
    "Time Management": "Prioritizing tasks to achieve efficiency and productivity",
    "Resilience": "Recovering quickly from setbacks and staying focused",
    "Problem-Solving": "Analyzing and resolving challenges effectively",
    "Crisis Management": "Leading effectively during high-stress situations"
}

CATEGORY_WEIGHTS = {"Strengths": 1.5, "Weaknesses": 1.2, "Opportunities": 1.4, "Threats": 1.1}
WATERMARK = "AI by Allam Rafi FKUI 2022"

# NLP Analysis
def analyze_text_with_confidence(text, qualities, confidence, category_weight):
    scores = {}
    explanations = {}
    for trait, description in qualities.items():
        trait_embedding = model.encode(description, convert_to_tensor=True)
        text_embedding = model.encode(text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(text_embedding, trait_embedding).item()
        weighted_score = similarity * (confidence / 10) * category_weight
        scores[trait] = weighted_score

        explanations[trait] = (
            f"Input: '{text}' aligns with '{trait}' ({description}). "
            f"Similarity: {similarity:.2f}, Confidence: {confidence}/10, Weighted Score: {weighted_score:.2f}."
        )
    return scores, explanations

# Generate Bar Chart
def generate_bar_chart(scores, category):
    plt.figure(figsize=(10, 6))
    traits = list(scores.keys())
    values = list(scores.values())
    plt.barh(traits, values, color=plt.cm.tab20.colors[:len(traits)])
    plt.xlabel("Scores")
    plt.ylabel("Traits")
    plt.title(f"{category} Analysis", fontsize=14)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()
    return buffer

# Generate Radar Chart
def generate_radar_chart(scores, category):
    traits = list(scores.keys())
    values = list(scores.values())
    values += values[:1]  # Close the radar chart
    angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticks([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(traits, fontsize=10)
    ax.set_title(f"{category} Radar Chart", fontsize=14)

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()
    return buffer

# Generate Heatmap
def generate_heatmap(scores):
    df = pd.DataFrame(scores).T
    fig = px.imshow(df, title="Trait Alignment Heatmap", color_continuous_scale="Viridis")
    return fig

# Leadership Viability Index (LSI)
def calculate_lsi(swot_scores, behavioral_score=0):
    strengths = sum([sum(scores.values()) for scores in swot_scores["Strengths"].values()])
    weaknesses = sum([sum(scores.values()) for scores in swot_scores["Weaknesses"].values()])
    opportunities = sum([sum(scores.values()) for scores in swot_scores["Opportunities"].values()])
    threats = sum([sum(scores.values()) for scores in swot_scores["Threats"].values()])

    # Normalize the categories to prevent bias
    total = strengths + weaknesses + opportunities + threats + 1e-9
    strengths /= total
    weaknesses /= total
    opportunities /= total
    threats /= total

    # Calculate LSI
    numerator = strengths + opportunities + behavioral_score
    denominator = weaknesses + threats + 1e-9
    lsi = np.log(numerator / denominator + 1e-9)
    return lsi

# Interpret LSI
def interpret_lsi(lsi):
    if lsi > 1.5:
        return "Exceptional Leadership Potential"
    elif lsi > 0.5:
        return "Good Leadership Potential"
    elif lsi > -0.5:
        return "Moderate Leadership Potential"
    else:
        return "Needs Improvement"

# Recommendations
def generate_recommendations(swot_explanations):
    recommendations = []
    for category, inputs in swot_explanations.items():
        if category in ["Weaknesses", "Threats"]:
            for text, explanations in inputs.items():
                for trait, explanation in explanations.items():
                    recommendations.append(f"Improve '{trait}': {explanation}")
    return recommendations

# Generate PDF Report
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'SWOT-Based Leadership Evaluation Report', align='C', ln=True)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f"Page {self.page_no()} - {WATERMARK}", align='C')

# Streamlit Interface
st.title("🌟 Advanced SWOT-Based Leadership Analysis 🌟")
st.sidebar.markdown(f"**{WATERMARK}**")

# Inputs
behavioral_responses = {q: st.text_area(q) for q in [
    "Describe how you handle stress.",
    "What motivates you to lead?",
]}

swot_inputs = {}
for category in ["Strengths", "Weaknesses", "Opportunities", "Threats"]:
    swot_inputs[category] = [
        (st.text_area(f"{category} #{i+1}"), st.slider(f"{category} #{i+1} Confidence", 1, 10, 5))
        for i in range(3)
    ]

if st.button("Analyze"):
    # NLP Analysis
    swot_scores, swot_explanations = {}, {}
    for category, inputs in swot_inputs.items():
        qualities = (
            LEADERSHIP_QUALITIES_POSITIVE if category in ["Strengths", "Opportunities"] else LEADERSHIP_QUALITIES_NEUTRAL
        )
        category_scores, category_explanations = {}, {}
        for text, confidence in inputs:
            if text.strip():
                scores, explanations = analyze_text_with_confidence(
                    text, qualities, confidence, CATEGORY_WEIGHTS[category]
                )
                category_scores[text] = scores
                category_explanations[text] = explanations
        swot_scores[category] = category_scores
        swot_explanations[category] = category_explanations

    # Calculate LSI
    lsi = calculate_lsi(swot_scores)
    lsi_interpretation = interpret_lsi(lsi)

    # Display Results
    st.subheader(f"Leadership Viability Index (LSI): {lsi:.2f}")
    st.write(f"**Interpretation**: {lsi_interpretation}")

    for category, traits in swot_scores.items():
        st.subheader(f"{category} Breakdown")
        for text, scores in traits.items():
            st.write(f"Input: {text}")
            for trait, score in scores.items():
                st.write(f"{trait}: {score:.2f}")

    # Visualizations
    st.plotly_chart(generate_heatmap(swot_scores))

    # Recommendations
    recommendations = generate_recommendations(swot_explanations)
    st.subheader("Recommendations")
    for rec in recommendations:
        st.write(f"- {rec}")

    # PDF Generation
    pdf = PDFReport()
    pdf.add_page()
    pdf.output("/tmp/report.pdf")
    st.download_button("Download Full Report", open("/tmp/report.pdf", "rb"), file_name="Leadership_Report.pdf", mime="application/pdf")
