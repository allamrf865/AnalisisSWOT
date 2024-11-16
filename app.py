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
BEHAVIORAL_QUESTIONS = [
    "Describe how you handle stressful situations.",
    "Explain your approach to team leadership.",
    "What motivates you to lead others?",
    "How do you make decisions under pressure?",
    "Describe a situation where you resolved a conflict."
]
WATERMARK = "AI by Allam Rafi FKUI 2022"

# Analyze behavioral inputs
def analyze_behavioral_responses(responses, qualities):
    scores = {}
    explanations = {}
    for question, response in responses.items():
        if response.strip():
            response_scores, response_explanations = analyze_text(response, qualities, 1.0)
            scores[question] = response_scores
            explanations[question] = response_explanations
    return scores, explanations

# Perform SWOT Analysis
def analyze_text(text, qualities, category_weight):
    scores = {}
    explanations = {}
    for trait, description in qualities.items():
        trait_embedding = model.encode(description, convert_to_tensor=True)
        text_embedding = model.encode(text, convert_to_tensor=True)

        similarity = util.pytorch_cos_sim(text_embedding, trait_embedding).item()
        weighted_score = similarity * category_weight
        scores[trait] = weighted_score

        explanation = (
            f"Input: '{text}' aligns with '{trait}' ({description}). "
            f"Similarity score: {similarity:.2f}. "
            f"Weighted score: {weighted_score:.2f}."
        )
        explanations[trait] = explanation
    return scores, explanations

# Visualizations
def create_radar_chart(scores, category):
    df = pd.DataFrame(scores.items(), columns=["Trait", "Score"])
    fig = px.line_polar(df, r="Score", theta="Trait", line_close=True, title=f"{category} Radar Chart")
    fig.update_traces(fill="toself")
    return fig

# PDF Report
class PDFReport(FPDF):
    # Implement the header, footer, and other sections as outlined previously
    pass

# Streamlit Application
st.title("🌟 Advanced SWOT-Based Leadership Analysis 🌟")

# Inputs for Behavioral and SWOT sections
behavioral_responses = {q: st.text_area(q) for q in BEHAVIORAL_QUESTIONS}
swot_inputs = {category: [st.text_area(f"{category} #{i+1}") for i in range(3)] for category in CATEGORY_WEIGHTS.keys()}

if st.button("Analyze"):
    # Analyze behavioral responses
    behavioral_scores, behavioral_explanations = analyze_behavioral_responses(behavioral_responses, LEADERSHIP_QUALITIES)

    # Perform SWOT analysis
    swot_scores = {}
    swot_explanations = {}
    for category, inputs in swot_inputs.items():
        category_scores = {}
        category_explanations = {}
        for text in inputs:
            if text.strip():
                scores, explanations = analyze_text(text, LEADERSHIP_QUALITIES, CATEGORY_WEIGHTS[category])
                category_scores[text] = scores
                category_explanations[text] = explanations
        swot_scores[category] = category_scores
        swot_explanations[category] = category_explanations

    # Display Results
    # (Insert logic to display LSI, SWOT breakdown, behavioral analysis, visualizations, etc.)

    # Generate PDF
    # (Insert logic to generate and download the PDF report)
