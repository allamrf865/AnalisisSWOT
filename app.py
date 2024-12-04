import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from fpdf import FPDF
from datetime import datetime

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
    - [IG](https://instagram.com/allamrf865)  
    - [Email](mailto:allamrafi@example.com)  
    """)
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

# Analyze Text with NLP
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

# Validate Inputs
def validate_inputs(swot_inputs, behavior_inputs):
    # Validate SWOT inputs including new categories
    for category, inputs in swot_inputs.items():
        for text, _ in inputs:
            if text.strip():  # Valid text
                return True
    # Validate behavioral analysis
    for response in behavior_inputs.values():
        if response.strip():
            return True
    return False  # If all inputs are empty

swot_inputs = {
    cat: [
        (
            st.text_area(f"{cat} #{i+1}", key=f"{cat}_text_{i+1}"),
            st.slider(f"{cat} #{i+1} Confidence", 1, 10, 5, key=f"{cat}_slider_{i+1}")
        )
        for i in range(3)
    ]
    for cat in ["Strengths", "Weaknesses", "Opportunities", "Threats", "External Opportunities", "External Threats"]
}

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)  # Bisa diubah ke DEBUG jika perlu lebih detail

def generate_bar_chart(data, output_path):
    categories = list(data.keys())
    values = []
    
    for cat in categories:
        try:
            # Pastikan data yang digunakan adalah numerik, jika tidak fallback ke 0
            if not all(isinstance(i, (int, float)) for i in data[cat]):
                raise ValueError(f"Non-numeric data found in category '{cat}'")
            mean_value = np.mean(data[cat]) if len(data[cat]) > 0 else 0
            values.append(mean_value)
        except (TypeError, ValueError) as e:
            # Catat peringatan dalam log, tanpa menampilkan peringatan UI
            values.append(0)
            logging.warning(f"Error in category '{cat}': {str(e)}. Defaulting to 0.")

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

# Generate 3D Scatter Plot with Error Handling
def generate_3d_scatter(data, output_path):
    # Ensure all categories have consistent keys and numeric values
    all_traits = {trait for traits in data.values() for trait in traits.keys()}
    fixed_data = {
        category: {trait: traits.get(trait, 0) for trait in all_traits}
        for category, traits in data.items()
    }

    try:
        # Prepare data for 3D scatter plot
        categories = list(fixed_data.keys())
        traits = list(all_traits)
        category_indices = np.arange(len(categories))

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        for category_idx, (category, traits_values) in enumerate(fixed_data.items()):
            xs = np.arange(len(traits_values))  # Traits indices
            ys = list(traits_values.values())  # Trait values
            zs = [category_idx] * len(xs)      # Fixed Z for each category
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
        st.warning("Ensure all input data is valid and numeric.")


# Generate 3D Surface Plot with Error Handling
def generate_3d_surface(data, output_path):
    # Ensure all categories have consistent keys and numeric values
    all_traits = {trait for traits in data.values() for trait in traits.keys()}
    fixed_data = {
        category: {trait: traits.get(trait, 0) for trait in all_traits}
        for category, traits in data.items()
    }
    
    try:
        # Convert to numpy array
        categories = list(fixed_data.keys())
        traits = list(all_traits)
        
        # Ensure consistent ordering of traits for each category
        z = np.array([list(fixed_data[category].values()) for category in categories])
        
        # Generate X and Y grids
        x = np.arange(len(categories))
        y = np.arange(len(traits))
        x, y = np.meshgrid(x, y)
        
        # Plot 3D Surface
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z.T, cmap="viridis", edgecolor='k')  # Transpose Z to match dimensions
        ax.set_title("3D Surface Plot - SWOT Scores")
        ax.set_xlabel("Categories")
        ax.set_ylabel("Traits")
        ax.set_zlabel("Scores")
        plt.savefig(output_path)
        plt.close()
    except ValueError as e:
        st.error(f"Error generating 3D surface plot: {e}")
        st.warning("Ensure data has consistent numeric values across all categories and traits.")

# Generate PDF Report
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, "SWOT-Based Leadership Evaluation Report", align='C', ln=True)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {WATERMARK}", align='C')

    def add_section(self, title, content):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=True)
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 10, content)

def generate_pdf_report(swot_scores, lsi, lsi_interpretation, behavior_results, chart_paths):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f"Leadership Viability Index (LSI): {lsi:.2f}", ln=True)
    pdf.cell(0, 10, f"Interpretation: {lsi_interpretation}", ln=True)

    # Add Behavior Results
    pdf.add_section("Behavioral Analysis Results", "\n".join([f"{q}: {a}" for q, a in behavior_results.items()]))

    # Add SWOT Scores
    for category, traits in swot_scores.items():
        pdf.add_section(f"{category} Scores", "\n".join([f"{trait}: {value:.2f}" for trait, value in traits.items()]))

    # Add Charts
    for chart_path in chart_paths:
        pdf.add_page()
        pdf.image(chart_path, x=10, y=50, w=190)
    pdf.output("/tmp/report.pdf")
    return "/tmp/report.pdf"

# Behavior Questions
behavior_questions = {
    "Q1": "Describe how you handle stress.",
    "Q2": "What motivates you to lead others?",
    "Q3": "How do you approach conflict resolution?",
    "Q4": "What is your strategy for long-term planning?",
    "Q5": "How do you inspire teamwork in challenging situations?"
}

# Collect Inputs
swot_inputs = {cat: [(st.text_area(f"{cat} #{i+1}"), st.slider(f"{cat} #{i+1} Confidence", 1, 10, 5)) for i in range(3)] for cat in ["Strengths", "Weaknesses", "Opportunities", "Threats"]}
behavior_inputs = {q: st.text_area(q) for q in behavior_questions.values()}

if st.button("Analyze"):
    if not validate_inputs(swot_inputs, behavior_inputs):
        st.warning("Please provide at least one valid input for SWOT or Behavioral analysis.")
    else:
        # Initialize swot_scores and swot_explanations here, inside the else block
        swot_scores, swot_explanations = {}, {}

        for category, inputs in swot_inputs.items():
            category_scores, category_explanations = {}, {}

            # Determine qualities based on category
            if category == "Strengths":
                qualities = LEADERSHIP_QUALITIES["Positive"]
            elif category == "Weaknesses":
                qualities = LEADERSHIP_QUALITIES["Negative"]
            elif category == "Opportunities":
                qualities = LEADERSHIP_QUALITIES["External Opportunities"]
            elif category == "Threats":
                qualities = LEADERSHIP_QUALITIES["External Threats"]
            else:
                qualities = LEADERSHIP_QUALITIES["Neutral"]

            # Process text and scores
            for text, confidence in inputs:
                scores, explanations = analyze_text_with_explanation(text, qualities, confidence, CATEGORY_WEIGHTS[category])
                category_scores.update(scores)
                category_explanations.update(explanations)

            # Save category scores and explanations
            swot_scores[category] = category_scores
            swot_explanations[category] = category_explanations

        # Analyze Behavior
        behavior_scores = {}
        for question, response in behavior_inputs.items():
            if response.strip():
                scores, _ = analyze_text_with_explanation(response, LEADERSHIP_QUALITIES["Positive"], 5, 1.0)
                behavior_scores[question] = scores

# Calculate LSI based on all SWOT categories
total_strengths = sum(swot_scores["Strengths"].values())
total_weaknesses = sum(swot_scores["Weaknesses"].values())
total_opportunities = sum(swot_scores["Opportunities"].values())
total_threats = sum(swot_scores["Threats"].values())

# Adjust formula to include all categories (Strengths, Weaknesses, Opportunities, and Threats)
lsi = np.log((total_strengths + total_opportunities + 1) / (total_weaknesses + total_threats + 1))

# Interpretation based on LSI value
lsi_interpretation = (
    "Exceptional Leadership Potential" if lsi > 1.5 else
    "Good Leadership Potential" if lsi > 0.5 else
    "Moderate Leadership Potential" if lsi > -0.5 else
    "Needs Improvement"
)

# Display LSI and Interpretation
st.subheader(f"Leadership Viability Index (LSI): {lsi:.2f}")
st.write(f"**Interpretation**: {lsi_interpretation}")

def generate_and_display_charts(swot_scores):
    # Define paths for saving charts
    heatmap_path = "/tmp/heatmap.png"
    scatter_chart_path = "/tmp/scatter_chart.png"
    surface_chart_path = "/tmp/surface_chart.png"
    
    # Generate charts
    generate_heatmap(swot_scores, heatmap_path)
    generate_3d_scatter(swot_scores, scatter_chart_path)
    generate_3d_surface(swot_scores, surface_chart_path)
    
    # Optionally, return paths if needed
    return heatmap_path, scatter_chart_path, surface_chart_path

# Now, outside the function, you display the images
heatmap_path, scatter_chart_path, surface_chart_path = generate_and_display_charts(swot_scores)

# Display the images using Streamlit
st.image(heatmap_path, caption="SWOT Heatmap")

# Display 3D Charts
st.image(scatter_chart_path, caption="3D Scatter Plot")
st.image(surface_chart_path, caption="3D Surface Plot")

        # Generate PDF Report
        pdf_path = generate_pdf_report(swot_scores, lsi, lsi_interpretation, behavior_inputs, [heatmap_path, scatter_chart_path, surface_chart_path])
        with open(pdf_path, "rb") as f:
            st.download_button("Download Professional PDF Report", f, "Leadership_Report.pdf", mime="application/pdf")

