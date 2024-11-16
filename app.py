import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import io

# Load multilingual NLP model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Leadership qualities for evaluation
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

# Function to calculate similarity scores
def calculate_scores(text, qualities, confidence, category_weight):
    scores = {}
    explanations = {}

    for trait, description in qualities.items():
        trait_embedding = model.encode(description, convert_to_tensor=True)
        text_embedding = model.encode(text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(text_embedding, trait_embedding).item()
        weighted_score = similarity * (confidence / 10) * category_weight
        scores[trait] = weighted_score

        explanation = f"Your input ('{text}') aligns with the trait '{trait}' ({description}). "
        if similarity > 0.7:
            explanation += f"Strong match with a similarity score of {similarity:.2f}."
        elif similarity > 0.4:
            explanation += f"Moderate match with a similarity score of {similarity:.2f}."
        else:
            explanation += f"Weak match with a similarity score of {similarity:.2f}."
        explanation += f" Final score adjusted by confidence ({confidence}/10) and category weight ({category_weight})."
        explanations[trait] = explanation

    return scores, explanations

# Function to create 2D visualizations
def create_visualizations(scores, category):
    if not scores:
        st.warning(f"No valid scores for {category}.")
        return None
    df = pd.DataFrame(list(scores.items()), columns=["Trait", "Score"]).sort_values(by="Score", ascending=False)
    fig = px.bar(
        df, x="Score", y="Trait", orientation="h", title=f"{category} Breakdown",
        color="Score", color_continuous_scale="Viridis"
    )
    fig.update_layout(xaxis_title="Score", yaxis_title="Traits", template="plotly_dark")
    return fig

# Function to create 3D visualizations
def create_3d_visualization(scores):
    if not scores:
        st.warning("No scores available for 3D visualization.")
        return None
    df = pd.DataFrame(scores.items(), columns=["Category", "Score"])
    df["Random Impact"] = np.random.uniform(0, 1, len(df))  # Random impact for visualization
    fig = go.Figure(data=[go.Scatter3d(
        x=df["Category"],
        y=df["Score"],
        z=df["Random Impact"],
        mode='markers',
        marker=dict(size=10, color=df["Score"], colorscale='Viridis', opacity=0.8)
    )])
    fig.update_layout(scene=dict(
        xaxis_title='Category',
        yaxis_title='Score',
        zaxis_title='Random Impact'
    ))
    return fig

# Leadership Viability Index (LSI)
def calculate_lsi(scores, behavioral_score, inconsistencies):
    epsilon = 1e-9
    positive = scores.get("Strengths", 0) + scores.get("Opportunities", 0) + behavioral_score
    negative = scores.get("Weaknesses", 0) + scores.get("Threats", 0)
    inconsistency_penalty = len(inconsistencies) * 0.1
    return np.log((positive / (negative + inconsistency_penalty + epsilon)) + epsilon)

# Interpret LSI
def interpret_lsi(lsi_score):
    if lsi_score > 1.5:
        return "Exceptional Leadership Potential. Highly suited for leadership roles."
    elif 1.0 < lsi_score <= 1.5:
        return "Strong Leadership Potential. Suitable for leadership but with minor areas for improvement."
    elif 0.5 < lsi_score <= 1.0:
        return "Moderate Leadership Potential. Notable areas for improvement exist."
    elif 0.0 < lsi_score <= 0.5:
        return "Low Leadership Potential. Requires significant development in key areas."
    else:
        return "Poor Leadership Fit. Needs substantial improvement before pursuing leadership roles."

# Streamlit App
st.title("🌟 Advanced SWOT-Based Leadership Analysis 🌟")
st.markdown(f"**Watermark:** {WATERMARK}")

# Input fields
swot_entries = {}
for category in ["Strengths", "Weaknesses", "Opportunities", "Threats"]:
    st.subheader(f"{category} Inputs")
    entries = []
    for i in range(3):
        text = st.text_area(f"{category} #{i + 1}", placeholder=f"Enter a {category} aspect...")
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
    explanations = {}
    for category, entries in swot_entries.items():
        breakdown = {}
        category_explanations = {}
        for text, confidence in entries:
            if text.strip():
                analysis, explanation = calculate_scores(text, LEADERSHIP_QUALITIES, confidence, CATEGORY_WEIGHTS[category])
                breakdown[text] = analysis
                category_explanations[text] = explanation
        swot_breakdown[category] = breakdown
        explanations[category] = category_explanations
        scores[category] = sum([sum(analysis.values()) for analysis in breakdown.values()]) if breakdown else 0

    behavioral_score = np.mean([len(example.split()) / 50 for example in behavioral_examples]) if behavioral_examples else 0
    lsi_score = calculate_lsi(scores, behavioral_score, [])

    st.metric("Leadership Viability Index (LSI)", f"{lsi_score:.2f}")
    st.markdown(f"**Interpretation:** {interpret_lsi(lsi_score)}")

    # Display visualizations and explanations
    for category, breakdown in swot_breakdown.items():
        st.subheader(f"{category} Breakdown")
        for text, analysis in breakdown.items():
            st.write(f"**Input**: {text}")
            for trait, score in analysis.items():
                st.write(f"- **{trait}**: {score:.2f}")
                st.write(f"  - Explanation: {explanations[category][text][trait]}")

        fig = create_visualizations(scores.get(category, {}), category)
        if fig:
            st.plotly_chart(fig)

    # Add 3D Visualization
    st.subheader("3D Visualization of SWOT Impact")
    fig_3d = create_3d_visualization(scores)
    if fig_3d:
        st.plotly_chart(fig_3d)
