import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd  

# ➡️ Configurer la page
st.set_page_config(page_title="Détecteur d'Émotions", page_icon="😊", layout="wide")

# ➡️ Custom CSS pour améliorer l'apparence
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stProgress > div > div > div > div {
        background-color: #ff4b4b;
    }
    </style>
""", unsafe_allow_html=True)

# ➡️ Définir les classes
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ➡️ Charger les métriques
confusion_matrix = np.load("confusion_matrix.npy")
model_accuracy = np.load("model_accuracy.npy")[0]

# ➡️ Charger et parser le rapport classification
with open("classification_report.txt", "r") as f:
    classification_report_text = f.read()

# 🔵 Fonction pour parser proprement en DataFrame
def parse_classification_report(text_report):
    lines = text_report.strip().split('\n')
    classes_local = []
    metrics = []

    for line in lines[2:-3]:  # On saute l'entête et les moyennes
        tokens = line.split()
        if len(tokens) == 5:
            classe, precision, recall, f1score, support = tokens
            classes_local.append(classe)
            metrics.append([float(precision), float(recall), float(f1score)])

    df = pd.DataFrame(metrics, index=classes_local, columns=['Precision', 'Recall', 'F1-Score'])
    return df

# Appliquer
report_df = parse_classification_report(classification_report_text)

# ➡️ Charger modèle sauvegardé
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 7)
    model.load_state_dict(torch.load("resnet18_fer2013.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ➡️ Fonction de prétraitement
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# ➡️ Fonction pour prédire
def predict(model, image_tensor):
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    return probabilities

# ➡️ Sidebar Informations modèle
with st.sidebar:
    st.title("🛠️ Paramètres du Modèle")
    st.metric(label="🎯 Accuracy Test", value=f"{model_accuracy*100:.2f}%")
    st.markdown("---")
    
    if st.button("📈 Voir la Matrice de Confusion"):
        st.subheader("📊 Matrice de Confusion :")
        fig_cm, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Prédictions')
        plt.ylabel('Vérités')
        st.pyplot(fig_cm)

    if st.button("📄 Voir le Rapport de Classification"):
        st.subheader("📄 Rapport Precision / Recall / F1-score :")
        st.dataframe(report_df.style.format("{:.2f}"))

    st.markdown("---")
    st.subheader("ℹ️ Infos Modèle")
    st.write("Modèle : **ResNet18 Fine-tuné**")
    st.write("Input : **Visage**")

# ➡️ Interface principale
st.title("😊 Détecteur Automatique d'Émotions Faciales")
st.write("Veuillez uploader une image de visage humain pour analyser l'émotion détectée.")

uploaded_file = st.file_uploader("📤 Upload votre image (formats : jpg, jpeg, png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns([1, 3])

    with col2:
        st.image(image, caption="🖼️ Image Chargée", width=400)

    with col1:
        if st.button('🚀 Analyser l\'émotion'):
            with st.spinner('🧠 Analyse de l\'émotion...'):
                time.sleep(1)
                image_tensor = preprocess_image(image)
                probabilities = predict(model, image_tensor)
                predicted_class = classes[torch.argmax(probabilities)]

            st.success(f"🌟 Émotion détectée : **{predicted_class.upper()}**")
            st.subheader("📊 Scores par émotion :")
            for idx, prob in enumerate(probabilities):
                st.progress(prob.item(), text=f"{classes[idx]} : {prob:.2%}")
