# Projet_Expressions_faciales-motions-
Données utilisées :
➔ FER2013 (images de visages annotés en 7 émotions : angry, disgust, fear, happy, neutral, sad, surprise)

Prétraitements appliqués :
➔ Redimensionnement (48x48)
➔ Conversion en niveaux de gris
➔ Normalisation
➔ Data Augmentation (rotation, translation, jitter)

Modèle Machine Learning :
➔ ResNet18 pré-entraîné (Transfer Learning)
➔ Adaptation de la 1ère couche pour images 1 canal (niveaux de gris)
➔ Fine-tuning avec scheduler (réduction du learning rate progressif)

Résultats obtenus :
Accuracy finale : ~62% sur set de test
Rapport Precision/Recall/F1-Score détaillé
Matrice de Confusion analysée
Difficulté : confusion principale entre sad / neutral / fear

Interface Utilisateur :
➔ Développée avec Streamlit.

➔ Fonctionnalités :
Upload d'image d'un visage
Prédiction de l'émotion avec scores


Visualisation des performances du modèle :
Accuracy
Matrice de Confusion stylisée
Tableau Precision / Recall / F1-score par classe


pip install -r requirements.txt

streamlit
torch
torchvision
numpy
matplotlib
seaborn
pandas
Pillow
scikit-learn

streamlit run app.py

