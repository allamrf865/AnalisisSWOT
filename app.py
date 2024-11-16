import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import entropy
from fpdf import FPDF  # For PDF report generation

# Load model with caching for speed optimization
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Leadership qualities and their descriptions
LEADERSHIP_QUALITIES = {
    "Leadership": "Ability to lead and inspire others",
    "Influence": "Capability to influence and motivate",
    "Vision": "Having a clear and inspiring vision",
    "Communication": "Effective communication skills",
    "Empathy": "Understanding and empathy towards others",
    "Strategic Thinking": "Ability to think strategically",
    "Conflict Resolution": "Skill in resolving conflicts effectively",
    "Resilience": "Ability to recover from setbacks and stay focused",
}

CATEGORY_WEIGHTS = {"Strengths": 1.5, "Weaknesses": 1.2, "Opportunities": 1.4, "Threats": 1.1}

WATERMARK = "AI by Allam Rafi FKUI 2022"

# Scoring function for inputs
def calculate_scores(text, qualities, confidence, category_weight, model):
    scores = {}
    for trait, description in qualities.items():
        trait_embedding = model.encode(description, convert_to_tensor=True)
        text_embedding = model.encode(text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(text_embedding, trait_embedding).item()
        weighted_score = similarity * (confidence / 10) * category_weight
        scores[trait] = weighted_score
    return scores

# Calculate entropy for diversity analysis
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

# Generate a PDF report
def generate_pdf_report(lsi_score, recommendations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="SWOT-Based Leadership Evaluation Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Leadership Viability Index (LSI): {lsi_score:.2f}", ln=True)
    pdf.ln(10)
    for rec in recommendations:
        pdf.multi_cell(0, 10, rec)
    pdf.ln(10)
    pdf.set_font("Arial", size=8, style="I")
    pdf.cell(0, 10, txt=WATERMARK, align="R")
    return pdf

# Main app
st.title("🌟 Comprehensive SWOT-Based Leadership Evaluation 🌟")
st.write("**Analyze your leadership potential with advanced AI insights.**")
st.markdown(f"**Watermark:** {WATERMARK}")

# Step 1: Context Selection
st.subheader("1️⃣ Select Leadership Context")
context = st.selectbox("Choose your leadership context:", ["Startup", "Corporate", "Educational", "General"])
CATEGORY_WEIGHTS.update({k: CATEGORY_WEIGHTS[k] + (0.1 if context == "Startup" and k == "Opportunities" else 0) for k in CATEGORY_WEIGHTS})

# Step 2: Input SWOT Descriptions
st.subheader("2️⃣ Input Your SWOT Analysis")
swot_entries = {}
for category in ["Strengths", "Weaknesses", "Opportunities", "Threats"]:
    st.write(f"### {category}")
    entries = []
    for i in range(3):
        text = st.text_area(f"{category} #{i + 1}", placeholder=f"Enter a {category} aspect...")
        confidence = st.slider(f"Confidence for {category} #{i + 1}", 1, 10, 5)
        entries.append((text, confidence))
    swot_entries[category] = entries

# Step 3: Behavioral Evidence
st.subheader("3️⃣ Provide Behavioral Evidence")
behavioral_examples = []
for i in range(2):
    example = st.text_area(f"Example #{i + 1}", placeholder="Describe a specific situation where you demonstrated leadership...")
    if example:
        behavioral_examples.append(example)

# Step 4: Analyze Results
if st.button("Analyze"):
    scores = {}
    entropy_scores = {}
    behavioral_score = np.mean([min(len(example.split()) / 50, 1.0) for example in behavioral_examples]) if behavioral_examples else 0

    # Calculate scores
    for category, entries in swot_entries.items():
        combined_text = " ".join([entry[0] for entry in entries if entry[0]])
        avg_conf = np.mean([entry[1] for entry in entries]) if entries else 5
        category_weight = CATEGORY_WEIGHTS[category]
        category_scores = calculate_scores(combined_text, LEADERSHIP_QUALITIES, avg_conf, category_weight, model)
        scores[category] = sum(category_scores.values())
        entropy_scores[category] = calculate_entropy(category_scores)

    # Calculate Leadership Viability Index
    inconsistencies = []  # Placeholder for any future inconsistency checks
    lsi_score = calculate_lsi(scores, behavioral_score, inconsistencies)

    # Display Results
    st.metric(label="Leadership Viability Index (LSI)", value=f"{lsi_score:.2f}")
    st.write(f"Behavioral Credibility Score: {behavioral_score:.2f}")

    # Recommendations
    st.subheader("Recommendations")
    recommendations = []
    for category, score in scores.items():
        if category in ["Strengths", "Opportunities"] and score < 50:
            recommendations.append(f"Improve your {category} by focusing on traits like Communication or Strategic Thinking.")
        elif category in ["Weaknesses", "Threats"] and score > 70:
            recommendations.append(f"Mitigate {category} issues by addressing areas like Conflict Avoidance or Procrastination.")
    for rec in recommendations:
        st.write(f"- {rec}")

    # Visualizations
    st.subheader("Visualizations")
    radar_data = pd.DataFrame.from_dict(scores, orient='index', columns=["Score"])
    radar_data["Category"] = radar_data.index

    # Radar Chart
    fig_radar = px.line_polar(radar_data, r="Score", theta="Category", line_close=True, title="SWOT Radar Chart")
    fig_radar.update_traces(fill="toself")
    st.plotly_chart(fig_radar)

    # 3D Scatter Plot
    fig_scatter = go.Figure(data=[go.Scatter3d(
        x=[scores["Strengths"]], y=[scores["Weaknesses"]], z=[scores["Opportunities"]],
        mode="markers", marker=dict(size=10, color=scores["Threats"], colorscale="Viridis")
    )])
    fig_scatter.update_layout(scene=dict(xaxis_title="Strengths", yaxis_title="Weaknesses", zaxis_title="Opportunities"))
    st.plotly_chart(fig_scatter)

    # PDF Report
    st.subheader("Download Report")
    pdf = generate_pdf_report(lsi_score, recommendations)
    pdf_file = f"/tmp/Leadership_Evaluation_Report.pdf"
    pdf.output(pdf_file)
    with open(pdf_file, "rb") as f:
        st.download_button("Download Report as PDF", f, file_name="Leadership_Evaluation_Report.pdf")

