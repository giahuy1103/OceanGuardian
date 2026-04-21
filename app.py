import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import os
import warnings

# Import model architecture
from src.model import CoralFusionModel

warnings.filterwarnings("ignore", category=FutureWarning)

print("Starting OceanGuardian Pro Web App...")

# ==========================================
# 1. SETUP & LOAD MODEL
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = 'data/processed/best_model.pth'

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please run train.py first!")

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
env_stats = checkpoint['env_stats']

model = CoralFusionModel(num_classes=2).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. INFERENCE & INSIGHT GENERATOR
# ==========================================
def predict_bleaching(image, sst, ph, lat, lon, month, heatwave):
    if image is None:
        return {"Please upload an image": 1.0}, "Waiting for visual data input..."
    
    # Process Image
    img_pil = Image.fromarray(image).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    # Process Tabular Data
    sst_norm = (sst - env_stats['sst_mean']) / (env_stats['sst_std'] + 1e-8)
    ph_norm = (ph - env_stats['ph_mean']) / (env_stats['ph_std'] + 1e-8)
    lat_norm = (lat - env_stats['lat_mean']) / (env_stats['lat_std'] + 1e-8)
    lon_norm = (lon - env_stats['lon_mean']) / (env_stats['lon_std'] + 1e-8)
    month_norm = (month - env_stats['month_mean']) / (env_stats['month_std'] + 1e-8)
    hw_val = 1.0 if heatwave else 0.0
    
    env_tensor = torch.tensor([[sst_norm, ph_norm, lat_norm, lon_norm, month_norm, hw_val]], dtype=torch.float32).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor, env_tensor)
        probs = torch.softmax(output, dim=1)[0]
        
    prob_healthy = probs[0].item()
    prob_bleached = probs[1].item()
    
    # Generate Scientific Insight
    insight = "**Ecosystem Status:** The coral reef exhibits healthy visual characteristics."
    if prob_bleached > 0.5:
        insight = "**Warning:** High probability of coral bleaching detected."
        if heatwave or sst > 30.0:
            insight += "\n\n *Insight:* Visual bleaching markers strongly correlate with the elevated Sea Surface Temperature (SST) and thermal stress in this region."
        if ph < 8.0:
            insight += "\n\n *Insight:* Low pH levels indicate ocean acidification, which degrades the coral's calcium carbonate skeleton, exacerbating the visual bleaching observed."

    return {"Healthy Coral": prob_healthy, "Bleached Coral": prob_bleached}, insight

# ==========================================
# 3. PRO UI DESIGN (GRADIO BLOCKS)
# ==========================================
# Custom Theme: Modern, Ocean-inspired colors with professional typography
ocean_theme = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="blue",
    font=[gr.themes.GoogleFont("Montserrat"), "sans-serif"]
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    block_title_text_weight="600",
    block_border_width="1px",
    block_shadow="*shadow_drop_sm"
)

with gr.Blocks(theme=ocean_theme, title="OceanGuardian AI") as demo:
    
    # Header Section
    with gr.Row():
        gr.Markdown(
            """
            # OceanGuardian: Multimodal Coral Bleaching Predictor
            **Ocean Futures Fellowship Prototype** | Powered by Early-Fusion Deep Learning (ResNet18 + Tabular MLP)
            
            Upload benthic imagery and input real-time oceanographic sensor data to assess coral health.
            """
        )
    
    # Main Dashboard
    with gr.Row():
        # LEFT COLUMN: INPUTS
        with gr.Column(scale=4):
            gr.Markdown("### 1. Visual Data (Benthic Survey)")
            img_input = gr.Image(label="Upload Coral Image", type="numpy", height=300)
            
            # Use Accordion to keep UI clean but accessible
            with gr.Accordion(" 2. Oceanographic Sensor Data", open=True):
                with gr.Row():
                    sst_input = gr.Slider(minimum=20.0, maximum=35.0, value=28.5, step=0.1, label="Sea Surface Temp (SST °C)")
                    ph_input = gr.Slider(minimum=7.5, maximum=8.5, value=8.1, step=0.01, label="Ocean pH Level")
                
                with gr.Row():
                    lat_input = gr.Number(value=-18.28, label="Latitude")
                    lon_input = gr.Number(value=147.69, label="Longitude")
                    month_input = gr.Dropdown(choices=[1,2,3,4,5,6,7,8,9,10,11,12], value=6, label="Observation Month")
                
                hw_input = gr.Checkbox(label=" Active Marine Heatwave Detected?", value=False)
                
            predict_btn = gr.Button(" Analyze Ecosystem Health", variant="primary", size="lg")
            
        # RIGHT COLUMN: OUTPUTS
        with gr.Column(scale=3):
            gr.Markdown("### Diagnostic Results")
            output_label = gr.Label(label="Prediction Probability")
            
            gr.Markdown("### Scientific Insights")
            output_insight = gr.Markdown(
                "Waiting for analysis...\n\n*The model will interpret the correlation between visual stress markers and environmental data here.*",
                elem_classes=["insight-box"]
            )
            
    # Footer & Technical Details
    with gr.Accordion(" System Architecture & Methodology", open=False):
        gr.Markdown(
            """
            - **Visual Branch:** ResNet18 extracts morphological features from reef imagery.
            - **Environmental Branch:** A Multi-Layer Perceptron (MLP) processes tabular climate data with Z-score normalization.
            - **Fusion Strategy:** Early concatenation with BatchNorm synchronization to balance modalities.
            - **Data Integrity:** Strict Train/Val stratification applied prior to mapping to eliminate Data Leakage.
            """
        )

    # Event trigger
    predict_btn.click(
        fn=predict_bleaching, 
        inputs=[img_input, sst_input, ph_input, lat_input, lon_input, month_input, hw_input], 
        outputs=[output_label, output_insight]
    )

if __name__ == "__main__":
    demo.launch(share=False)