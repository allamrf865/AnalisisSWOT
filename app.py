import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fpdf import FPDF
from datetime import datetime
import logging

# Streamlit Config
st.set_page_config(page_title="Advanced SWOT Leadership Analysis", page_icon="🌟", layout="wide")

# Define Watermark
WATERMARK = "AI by Muhammad Allam Rafi, FKUI 2022"

# Load NLP Model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Define Leadership Traits
LEADERSHIP_QUALITIES = {
    "Positive": {
        "Leadership": "Ability to lead and inspire others (Internal Strength).",
        "Vision": "Clear and inspiring direction for the future (Internal Strength).",
        "Integrity": "Acting consistently with honesty and strong ethics (Internal Strength).",
        "Innovation": "Driving creativity and fostering change (Internal Strength).",
        "Inclusivity": "Promoting diversity and creating an inclusive environment (Internal Strength).",
        "Empathy": "Understanding others' perspectives and feelings (Internal Strength).",
        "Communication": "Conveying ideas clearly and effectively (Internal Strength).",
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
    },
    "External Opportunities": {
        "Emerging Markets": "New geographical or sectoral markets that offer growth potential.",
        "Technological Advancements": "Technological developments that can be leveraged for innovation.",
        "Regulatory Changes": "Changes in regulations that create opportunities for growth or market entry.",
        "Partnership Opportunities": "Potential collaborations with other organizations for mutual benefit.",
        "Globalization": "Access to international markets and partnerships as a result of global interconnectedness."
    },
    "External Threats": {
        "Academic Pressure": "Challenges arising from academic performance or deadlines.",
        "Extracurricular Commitments": "Overinvolvement in external activities that could detract from leadership focus.",
        "Economic Downturn": "External economic conditions affecting organizational success.",
        "Technological Disruptions": "External technological advancements that might reduce the relevance of current leadership strategies.",
        "Market Competition": "Increased competition in the job market or in a specific field."
    }
}


CATEGORY_WEIGHTS = {"Strengths": 1.5, "Weaknesses": 1.3, "Opportunities": 1.4, "Threats": 1.2}

# Sidebar Information
with st.sidebar:
    st.image("https://via.placeholder.com/150", caption="Muhammad Allam Rafi", use_column_width=True)
    st.markdown("### **🌟 About Me 🌟**")
    st.markdown("""👨‍⚕️ **Medical Student**\n Passionate about **Machine Learning**, **Leadership Research**, and **Healthcare AI**.""")
    st.markdown("### **📫 Contact Me**")
    st.markdown("[IG](https://instagram.com/allamrf865)\n[Email](mailto:allamrafi@example.com)")
    st.markdown(f"---\n🌟 **{WATERMARK}** 🌟")

# Header Section
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌟 SWOT Leadership Analysis 🌟</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #808080;'>Advanced Evaluation of Leadership Potential</h3>", unsafe_allow_html=True)
st.markdown("---")

# Dynamic NLP Explanation Function
def dynamic_nlp_explanation(text, traits):
    text_embedding = model.encode(text, convert_to_tensor=True)
    explanations = []
    for trait, description in traits.items():
        trait_embedding = model.encode(description, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(text_embedding, trait_embedding).item()
        if similarity > 0.75:
            relevance = "highly aligns"
        elif similarity > 0.5:
            relevance = "moderately aligns"
        else:
            relevance = "has low alignment"
        explanations.append(
            f"'{text}' {relevance} with the trait '{trait}' ({description}). Similarity score: {similarity:.2f}."
        )
    return explanations

# Validate Inputs
def validate_inputs(swot_inputs, behavior_inputs):
    for category, inputs in swot_inputs.items():
        for text, _ in inputs:
            if text.strip():
                return True
    for response in behavior_inputs.values():
        if response.strip():
            return True
    return False

# Analyze Text with Explanation
def analyze_text_with_explanation(text, qualities, confidence, category_weight):
    if not text.strip():
        return {}, {}
    scores, explanations = {}, {}
    embeddings = model.encode([text] + list(qualities.values()), convert_to_tensor=True)
    text_embedding, trait_embeddings = embeddings[0], embeddings[1:]
    similarities = util.pytorch_cos_sim(text_embedding, trait_embeddings).squeeze().tolist()
    
    for trait, similarity in zip(qualities.keys(), similarities):
        weighted_score = similarity * (confidence / 10) * category_weight
        scores[trait] = weighted_score
        explanations[trait] = f"Input aligns with '{trait}'. Similarity: {similarity:.2f}, Weighted Score: {weighted_score:.2f}."
    return scores, explanations

# Generate Charts
def generate_and_display_charts(swot_scores):
    # Generate Heatmap
    def generate_heatmap(data, output_path):
        df = pd.DataFrame(data).T.fillna(0)
        plt.figure(figsize=(10, 6))
        plt.imshow(df, cmap="viridis", interpolation="nearest", aspect="auto")
        plt.colorbar(label="Scores")
        plt.title("SWOT Heatmap")
        plt.xlabel("Traits")
        plt.ylabel("Categories")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    # Generate 3D Scatter Plot
    def generate_3d_scatter(data, output_path):
        all_traits = {trait for traits in data.values() for trait in traits.keys()}
        fixed_data = {category: {trait: traits.get(trait, 0) for trait in all_traits} for category, traits in data.items()}

        try:
            categories = list(fixed_data.keys())
            traits = list(all_traits)
            category_indices = np.arange(len(categories))

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            for category_idx, (category, traits_values) in enumerate(fixed_data.items()):
                xs = np.arange(len(traits_values))
                ys = list(traits_values.values())
                zs = [category_idx] * len(xs)
                ax.scatter(xs, zs, ys, label=category)

            ax.set_title("3D Scatter Plot - SWOT Scores")
            ax.set_xlabel("Traits")
            ax.set_ylabel("Categories")
            ax.set_zlabel("Scores")
            ax.legend(loc="best")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            st.error(f"Error generating 3D scatter plot: {e}")

    # Generate 3D Surface Plot
    def generate_3d_surface(data, output_path):
        all_traits = {trait for traits in data.values() for trait in traits.keys()}
        fixed_data = {category: {trait: traits.get(trait, 0) for trait in all_traits} for category, traits in data.items()}
        
        try:
            categories = list(fixed_data.keys())
            traits = list(all_traits)
            z = np.array([list(fixed_data[category].values()) for category in categories])
            x = np.arange(len(categories))
            y = np.arange(len(traits))
            x, y = np.meshgrid(x, y)

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(x, y, z.T, cmap="viridis", edgecolor='k')
            ax.set_title("3D Surface Plot - SWOT Scores")
            ax.set_xlabel("Categories")
            ax.set_ylabel("Traits")
            ax.set_zlabel("Scores")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            st.error(f"Error generating 3D surface plot: {e}")

    # File paths for saving
    heatmap_path = "/mnt/data/heatmap.png"
    scatter_path = "/mnt/data/3d_scatter_plot.png"
    surface_path = "/mnt/data/3d_surface_plot.png"
    
    generate_heatmap(swot_scores, heatmap_path)
    generate_3d_scatter(swot_scores, scatter_path)
    generate_3d_surface(swot_scores, surface_path)

    return heatmap_path, scatter_path, surface_path

# Generate PDF Report
def generate_pdf_report(swot_scores, heatmap_path, scatter_path, surface_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="SWOT Leadership Analysis Report", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt="SWOT Analysis Summary:", ln=True)

    for category, score in swot_scores.items():
        pdf.cell(200, 10, txt=f"{category}: {score}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt="Generated Charts:", ln=True)
    pdf.image(heatmap_path, w=180)
    pdf.ln(60)
    pdf.image(scatter_path, w=180)
    pdf.ln(60)
    pdf.image(surface_path, w=180)

    # Save PDF to file
    output_pdf = f"/mnt/data/SWOT_Leadership_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(output_pdf)
    return output_pdf

# Main Logic
def main():
    st.title("SWOT Leadership Analysis")

    with st.form(key="swot_form"):
        strengths_input = st.text_area("Strengths")
        weaknesses_input = st.text_area("Weaknesses")
        opportunities_input = st.text_area("Opportunities")
        threats_input = st.text_area("Threats")
        behavior_response = st.text_area("Leadership Behavior Response")

        submit_button = st.form_submit_button("Analyze")

    if submit_button:
        if validate_inputs({
            "Strengths": [(strengths_input, "High")],
            "Weaknesses": [(weaknesses_input, "Low")],
            "Opportunities": [(opportunities_input, "High")],
            "Threats": [(threats_input, "Low")]
        }, behavior_response):
            swot_scores = {
                "Strengths": strengths_input,
                "Weaknesses": weaknesses_input,
                "Opportunities": opportunities_input,
                "Threats": threats_input
            }
            heatmap_path, scatter_path, surface_path = generate_and_display_charts(swot_scores)
            pdf_path = generate_pdf_report(swot_scores, heatmap_path, scatter_path, surface_path)

            st.image(heatmap_path, caption="Heatmap", use_column_width=True)
            st.image(scatter_path, caption="3D Scatter Plot", use_column_width=True)
            st.image(surface_path, caption="3D Surface Plot", use_column_width=True)

            st.markdown(f"[Download SWOT Report PDF](sandbox:{pdf_path})")
