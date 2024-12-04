import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import shap
from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit Configuration
st.set_page_config(page_title="SWOT Leadership Analysis", page_icon="🌟", layout="wide")

# Define Watermark
WATERMARK = "Advanced AI Leadership Analysis by Muhammad Allam Rafi, CBOA® CDSP®"

# Load Transformer Model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Define Leadership Traits for SWOT
LEADERSHIP_TRAITS = {
    "Positive": {
        "Leadership": "Ability to lead and inspire others.",
        "Vision": "Clear and inspiring direction for the future.",
        "Integrity": "Acting consistently with honesty and strong ethics.",
        "Innovation": "Driving creativity and fostering change.",
        "Inclusivity": "Promoting diversity and creating an inclusive environment.",
        "Empathy": "Understanding others' perspectives and feelings.",
        "Communication": "Conveying ideas clearly and effectively.",
    },
    "Neutral": {
        "Adaptability": "Flexibility to adjust to new challenges.",
        "Time Management": "Prioritizing and organizing tasks efficiently.",
        "Problem-Solving": "Resolving issues effectively.",
        "Conflict Resolution": "Managing disagreements constructively.",
        "Resilience": "Bouncing back from setbacks.",
    },
    "Negative": {
        "Micromanagement": "Excessive control over tasks.",
        "Overconfidence": "Ignoring input due to arrogance.",
        "Conflict Avoidance": "Avoiding necessary confrontations.",
        "Indecisiveness": "Inability to make timely decisions.",
        "Rigidity": "Refusing to adapt to new circumstances.",
    }
}

CATEGORY_WEIGHTS = {"Strengths": 1.5, "Weaknesses": 1.3, "Opportunities": 1.4, "Threats": 1.2}

# Sidebar Profile and Contact
with st.sidebar:
    st.image("https://via.placeholder.com/150", caption="Muhammad Allam Rafi", use_column_width=True)
    st.markdown("### **🌟 About Me 🌟**")
    st.markdown("""
    👨‍⚕️ **Medical Student**  
    Passionate about **Machine Learning**, **Leadership Research**, and **Healthcare AI**.  
    - 🎓 **Faculty of Medicine**, Universitas Indonesia  
    - 📊 **Research Interests**:  
      - Leadership Viability in Healthcare  
      - AI-driven solutions for medical challenges  
      - Natural Language Processing and Behavioral Analysis  
    - 🧑‍💻 **Skills**: Python, NLP, Data Visualization
    """)
    st.markdown("### **📫 Contact Me**")
    st.markdown("""
    - [LinkedIn](https://linkedin.com)  
    - [GitHub](https://github.com)  
    - [Email](mailto:allamrafi@example.com)  
    """)
    st.markdown(f"---\n🌟 **{WATERMARK}** 🌟")

# Header Section
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌟 Advanced SWOT Leadership Analysis 🌟</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #808080;'>Discover Leadership Potential with Explainable AI</h3>", unsafe_allow_html=True)
st.markdown("---")

# NLP Analysis Function
def analyze_text_with_shap(text, traits, confidence, category_weight):
    """Analyze text using SHAP-like explanations."""
    if not text.strip():
        return {}, {}

    scores, explanations = {}, {}
    text_embedding = model.encode(text, convert_to_tensor=True)
    trait_embeddings = model.encode(list(traits.values()), convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(text_embedding, trait_embeddings).squeeze().tolist()

    for trait, similarity in zip(traits.keys(), similarities):
        weighted_score = similarity * (confidence / 10) * category_weight
        scores[trait] = max(0, weighted_score)  # Ensure scores are non-negative
        explanations[trait] = f"'{text}' aligns with '{trait}'. Similarity: {similarity:.2f}, Weighted Score: {weighted_score:.2f}."

    return scores, explanations

# Input Validation
def validate_swot_inputs(swot_inputs):
    for category, entries in swot_inputs.items():
        for text, _ in entries:
            if text.strip():
                return True
    return False

def validate_behavioral_inputs(behavior_inputs):
    for response in behavior_inputs.values():
        if response.strip():
            return True
    return False

# Input Fields
swot_inputs = {
    category: [
        (st.text_area(f"{category} #{i+1}", key=f"{category}_{i}"), 
         st.slider(f"{category} #{i+1} Confidence", 1, 10, 5, key=f"{category}_confidence_{i}"))
        for i in range(3)
    ]
    for category in ["Strengths", "Weaknesses", "Opportunities", "Threats"]
}

behavior_questions = {
    "Q1": "Describe how you handle stress.",
    "Q2": "What motivates you to lead others?",
    "Q3": "How do you approach conflict resolution?",
    "Q4": "What is your strategy for long-term planning?",
    "Q5": "How do you inspire teamwork in challenging situations?"
}
behavior_responses = {q: st.text_area(q, key=f"behavior_{i}") for i, q in enumerate(behavior_questions.values())}

# 3D Scatter Plot
def generate_3d_scatter(data):
    """Generates 3D scatter plot."""
    fig = go.Figure()
    categories = list(data.keys())
    for idx, (category, traits) in enumerate(data.items()):
        xs = list(range(len(traits)))
        ys = [idx] * len(traits)
        zs = list(traits.values())

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers',
            marker=dict(size=5, color=zs, colorscale='Viridis'),
            name=category
        ))

    fig.update_layout(
        title="3D Scatter Plot - SWOT Analysis",
        scene=dict(xaxis_title="Traits", yaxis_title="Categories", zaxis_title="Scores")
    )
    return fig

# Advanced Behavioral Analysis (Meta-Learning Inspired)
def behavioral_meta_learning(behavior_responses, swot_scores):
    """
    Adapt behavioral scoring dynamically based on patterns from SWOT scores.
    """
    behavior_scores = {}
    for question, response in behavior_responses.items():
        if response.strip():
            # Heuristic: Use average of Strengths and Opportunities to adapt behavioral scoring
            strengths_avg = np.mean(list(swot_scores.get("Strengths", {}).values()) or [0])
            opportunities_avg = np.mean(list(swot_scores.get("Opportunities", {}).values()) or [0])
            behavior_scores[question] = (strengths_avg + opportunities_avg) * 0.7  # Adjusted weight
    return behavior_scores

# Advanced Machine Learning: KMeans Clustering for SWOT
def kmeans_clustering(data):
    """
    Perform clustering on SWOT traits for enhanced visualization.
    """
    all_traits = []
    trait_labels = []
    for category, traits in data.items():
        all_traits.extend(list(traits.values()))
        trait_labels.extend([f"{category}-{trait}" for trait in traits.keys()])

    # Reshape for clustering
    traits_array = np.array(all_traits).reshape(-1, 1)

    # Perform KMeans clustering
    n_clusters = min(len(traits_array), 3)  # Limit clusters to avoid over-segmentation
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(traits_array)

    # Create dataframe for visualization
    cluster_df = pd.DataFrame({
        "Trait": trait_labels,
        "Score": all_traits,
        "Cluster": kmeans.labels_
    })
    return cluster_df

# Advanced Visualization: SHAP for Explainability
def visualize_shap_values(traits, shap_values):
    """
    Generate a bar chart for SHAP values to explain trait alignment.
    """
    st.markdown("### SHAP Explanation: Alignment of Input with Traits")
    fig = go.Figure(go.Bar(
        x=shap_values,
        y=traits,
        orientation='h',
        marker=dict(color='blue'),
    ))
    fig.update_layout(
        title="Explainable Alignment with Leadership Traits",
        xaxis_title="SHAP Value",
        yaxis_title="Traits"
    )
    st.plotly_chart(fig, use_container_width=True)

# Advanced Visualization: Interactive 3D Surface
def generate_3d_surface(data):
    """
    Generates an interactive 3D surface plot using Plotly.
    """
    categories = list(data.keys())
    traits = list(next(iter(data.values())).keys())
    z = np.array([list(traits.values()) for traits in data.values()])
    x, y = np.meshgrid(range(len(categories)), range(len(traits)))

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
    fig.update_layout(
        title="3D Surface Plot: SWOT Scores",
        scene=dict(
            xaxis=dict(title="Categories"),
            yaxis=dict(title="Traits"),
            zaxis=dict(title="Scores"),
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# Generate Advanced PDF Report
class AdvancedPDF(FPDF):
    """
    Custom PDF class for generating professional SWOT Leadership reports.
    """
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, "Advanced SWOT Leadership Analysis Report", align='C', ln=True)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {WATERMARK}", align='C')

    def add_section(self, title, content):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=True)
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 10, content)

def generate_pdf_report(swot_scores, lsi, lsi_interpretation, behavior_scores, cluster_df, chart_paths):
    """
    Generate PDF with SWOT analysis results, behavioral analysis, and visualizations.
    """
    pdf = AdvancedPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f"Leadership Viability Index (LSI): {lsi:.2f}", ln=True)
    pdf.cell(0, 10, f"Interpretation: {lsi_interpretation}", ln=True)

    # Add Behavioral Analysis
    pdf.add_section("Behavioral Analysis", "\n".join([f"{q}: {s:.2f}" for q, s in behavior_scores.items()]))

    # Add SWOT Scores and Clusters
    for category, traits in swot_scores.items():
        pdf.add_section(f"{category} Scores", "\n".join([f"{trait}: {value:.2f}" for trait, value in traits.items()]))
    pdf.add_section("KMeans Clustering", cluster_df.to_string(index=False))

    # Add Charts
    for chart_path in chart_paths:
        pdf.add_page()
        pdf.image(chart_path, x=10, y=50, w=190)
    pdf.output("/tmp/advanced_report.pdf")
    return "/tmp/advanced_report.pdf"

# Execution: Perform Full Analysis
if st.button("Analyze"):
    if not (validate_swot_inputs(swot_inputs) or validate_behavioral_inputs(behavior_responses)):
        st.error("Please provide at least one valid SWOT input or Behavioral response.")
    else:
        st.success("Analysis in progress...")

        # Process SWOT Inputs
        swot_scores = {}
        for category, entries in swot_inputs.items():
            qualities = (
                LEADERSHIP_TRAITS["Positive"] if category in ["Strengths", "Opportunities"] else
                LEADERSHIP_TRAITS["Negative"] if category == "Threats" else
                LEADERSHIP_TRAITS["Neutral"]
            )
            category_scores = {}
            for text, confidence in entries:
                if text.strip():
                    scores, _ = analyze_text_with_shap(text, qualities, confidence, CATEGORY_WEIGHTS[category])
                    category_scores.update(scores)
            swot_scores[category] = category_scores or {trait: 0 for trait in qualities.keys()}

        # Behavioral Analysis
        behavior_scores = behavioral_meta_learning(behavior_responses, swot_scores)

        # LSI Calculation
        total_strengths = sum(swot_scores.get("Strengths", {}).values())
        total_weaknesses = sum(swot_scores.get("Weaknesses", {}).values())
        lsi = np.log((total_strengths + 1) / (total_weaknesses + 1))
        lsi_interpretation = (
            "Exceptional Leadership Potential" if lsi > 1.5 else
            "Good Leadership Potential" if lsi > 0.5 else
            "Moderate Leadership Potential" if lsi > -0.5 else
            "Needs Improvement"
        )

        # Display Results
        st.subheader(f"Leadership Viability Index (LSI): {lsi:.2f}")
        st.write(f"**Interpretation**: {lsi_interpretation}")

        # Generate Visualizations
        st.markdown("### SWOT 2D and 3D Visualizations")
        generate_3d_surface(swot_scores)

        # KMeans Clustering
        cluster_df = kmeans_clustering(swot_scores)
        st.write("### SWOT Clustering Results")
        st.dataframe(cluster_df)

        # Generate PDF
        st.markdown("### Generate Report")
        pdf_path = generate_pdf_report(swot_scores, lsi, lsi_interpretation, behavior_scores, cluster_df, [])
        with open(pdf_path, "rb") as pdf_file:
            st.download_button("Download Full Report", pdf_file, "SWOT_Report.pdf", mime="application/pdf")

import torch
import torch.nn as nn
from transformers import pipeline

# Define a Deep Learning Model for Regression
class LeadershipPotentialModel(nn.Module):
    def __init__(self, input_dim):
        super(LeadershipPotentialModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Train Predictive Model
def train_regression_model(data, targets):
    """
    Train a simple regression model to predict Leadership Potential Score.
    """
    input_dim = data.shape[1]
    model = LeadershipPotentialModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Convert data to PyTorch tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    # Training loop
    for epoch in range(200):  # Iterasi 200 epoch
        optimizer.zero_grad()
        outputs = model(data_tensor)
        loss = criterion(outputs, targets_tensor)
        loss.backward()
        optimizer.step()

    return model

# Generate Auto-Recommendations
def generate_recommendations(swot_scores):
    """
    Generate actionable recommendations based on SWOT scores.
    """
    recommendations = []
    for category, traits in swot_scores.items():
        for trait, score in traits.items():
            if category == "Strengths" and score > 0.7:
                recommendations.append(f"Leverage your strength in '{trait}' to inspire others.")
            elif category == "Weaknesses" and score > 0.5:
                recommendations.append(f"Work on improving your '{trait}' to minimize its impact.")
            elif category == "Opportunities" and score > 0.6:
                recommendations.append(f"Explore opportunities related to '{trait}' to grow.")
            elif category == "Threats" and score > 0.4:
                recommendations.append(f"Mitigate threats in '{trait}' to ensure sustainability.")
    return recommendations

# Advanced Sentiment Analysis
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_sentiment_model()

def analyze_sentiment(swot_inputs):
    """
    Perform sentiment analysis on SWOT inputs.
    """
    sentiments = {}
    for category, entries in swot_inputs.items():
        sentiments[category] = []
        for text, _ in entries:
            if text.strip():
                sentiment_result = sentiment_model(text[:512])  # Limit text length to 512 chars
                sentiments[category].append(sentiment_result[0])
            else:
                sentiments[category].append({"label": "Neutral", "score": 0.0})
    return sentiments

# 3D Time-Series Visualization
def generate_3d_time_series(data, time_range):
    """
    Generate a 3D time-series plot showing the evolution of SWOT scores.
    """
    fig = go.Figure()

    categories = list(data.keys())
    time_steps = range(len(time_range))
    for i, category in enumerate(categories):
        for j, (trait, scores) in enumerate(data[category].items()):
            fig.add_trace(go.Scatter3d(
                x=time_steps, 
                y=[i] * len(time_steps), 
                z=scores,
                mode='lines',
                name=f"{category} - {trait}"
            ))

    fig.update_layout(
        title="3D Time-Series SWOT Visualization",
        scene=dict(
            xaxis=dict(title="Time"),
            yaxis=dict(title="Categories"),
            zaxis=dict(title="Scores")
        )
    )
    return fig

# Behavioral Clustering Analysis
def cluster_behavioral_responses(behavior_responses):
    """
    Cluster behavioral responses using KMeans for pattern detection.
    """
    embeddings = []
    for response in behavior_responses.values():
        if response.strip():
            embedding = model.encode(response, convert_to_tensor=False)
            embeddings.append(embedding)

    if len(embeddings) > 1:
        kmeans = KMeans(n_clusters=min(3, len(embeddings)), random_state=42)
        labels = kmeans.fit_predict(embeddings)
        clustered_responses = {f"Cluster {label}": [] for label in set(labels)}
        for idx, label in enumerate(labels):
            clustered_responses[f"Cluster {label}"].append(list(behavior_responses.values())[idx])
    else:
        clustered_responses = {"Cluster 0": list(behavior_responses.values())}

    return clustered_responses
