"""
Interface Streamlit interactive pour tester et corriger le modèle de classification d'images de nourriture.
"""

import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.engine import load_checkpoint
from src.model import create_model, topk_predictions


def make_transform(image_size: int):
    """Crée la transformation d'image standard."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@st.cache_resource
def load_model(checkpoint_path: Path, image_size: int = 224):
    """Charge le modèle entraîné et le checkpoint."""
    device = torch.device("cpu")
    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = checkpoint["class_names"]

    model = create_model(
        checkpoint["backbone"],
        num_classes=len(class_names),
        pretrained=False
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, class_names, device, image_size


def predict(model, image_tensor, device, top_k=5):
    """Fait une prédiction sur une image."""
    with torch.no_grad():
        logits = model(image_tensor)
    return logits


def get_probabilities(logits, class_names, top_k=5):
    """Extrait les probabilités et les noms de classes."""
    scores = torch.softmax(logits, dim=1)[0]
    top_scores, top_indices = torch.topk(scores, min(top_k, len(class_names)))

    results = []
    for score, idx in zip(top_scores, top_indices):
        results.append({
            "class_name": class_names[idx.item()],
            "score": score.item(),
            "confidence": f"{score.item() * 100:.1f}%"
        })
    return results


def main():
    st.set_page_config(page_title="🍕 Food Classifier", layout="wide")
    st.title("🍕 Testeur d'IA - Classification d'Images de Nourriture")

    # Initialisation de la session
    if "feedback_log" not in st.session_state:
        st.session_state.feedback_log = []
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None

    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        checkpoint_path = Path(st.text_input(
            "Chemin du modèle",
            value="models/best.pt"
        ))

        top_k = st.slider("Afficher le top-K prédictions", 1, 10, 5)

        if not checkpoint_path.exists():
            st.error(f"❌ Le modèle n'existe pas: {checkpoint_path}")
            return

        st.divider()
        st.subheader("📊 Statistiques")
        st.write(f"Total feedback collecté: {len(st.session_state.feedback_log)}")

    # Charger le modèle
    try:
        model, class_names, device, image_size = load_model(checkpoint_path, image_size=224)
        st.success(f"✅ Modèle chargé! ({len(class_names)} classes)")
    except Exception as e:
        st.error(f"Erreur au chargement du modèle: {e}")
        return

    # Interface principale
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("📸 Charger une Image")

        # Options d'upload
        upload_option = st.radio("Choisir la source:", ["Upload depuis fichier", "Fichier exemple"])

        image = None

        if upload_option == "Upload depuis fichier":
            uploaded_file = st.file_uploader(
                "Sélectionnez une image",
                type=["jpg", "jpeg", "png", "gif", "bmp"]
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
        else:
            # Lister les dossiers de classes disponibles
            raw_dir = Path("data/raw")
            if raw_dir.exists():
                classes = [d.name for d in raw_dir.iterdir() if d.is_dir()]
                if classes:
                    selected_class = st.selectbox("Choisir une classe d'exemple:", sorted(classes))

                    class_path = raw_dir / selected_class
                    image_files = list(class_path.glob("*.*"))

                    if image_files:
                        selected_image = st.selectbox(
                            "Choisir une image:",
                            image_files,
                            format_func=lambda x: x.name
                        )
                        image = Image.open(selected_image).convert("RGB")
                    else:
                        st.warning(f"Aucune image trouvée dans {class_path}")
                else:
                    st.warning("Aucune classe trouvée dans data/raw")
            else:
                st.error(f"Dossier non trouvé: {raw_dir}")

        # Afficher l'image si elle a été chargée
        if image is not None:
            st.image(image, caption="Image chargée", use_column_width=True)
        else:
            st.info("Veuillez charger une image")

        # Bouton de prédiction
        if st.button("🔮 Faire une Prédiction", key="predict_btn", use_container_width=True):
            if image is None:
                st.error("❌ Veuillez d'abord charger une image!")
            else:
                # Transformer l'image
                transform = make_transform(image_size)
                image_tensor = transform(image).unsqueeze(0).to(device)

                # Faire la prédiction
                logits = predict(model, image_tensor, device)
                predictions = get_probabilities(logits, class_names, top_k)

                st.session_state.prediction_result = {
                    "predictions": predictions,
                    "image": image,
                }

    # Colonne droite - Résultats
    with col2:
        st.subheader("🎯 Résultats")

        if st.session_state.prediction_result is not None:
            predictions = st.session_state.prediction_result["predictions"]

            # Afficher le meilleur résultat en gros
            best_pred = predictions[0]
            st.metric(
                label="🏆 Meilleure prédiction",
                value=best_pred["class_name"].replace("_", " ").title(),
                delta=best_pred["confidence"]
            )

            # Afficher tous les résultats
            st.write("**Top prédictions:**")
            for i, pred in enumerate(predictions, 1):
                confidence = pred["score"] * 100
                bar_color = "🟢" if i == 1 else "🟡" if i <= 3 else "🔴"
                st.write(f"{bar_color} **{i}. {pred['class_name'].replace('_', ' ').title()}**: {confidence:.1f}%")

            # Section Feedback
            st.divider()
            st.subheader("✅ Feedback")

            feedback_col1, feedback_col2 = st.columns(2)

            with feedback_col1:
                if st.button("👍 C'est Correct!", key="correct_btn", use_container_width=True):
                    feedback_entry = {
                        "type": "correct",
                        "predicted_class": best_pred["class_name"],
                        "confidence": best_pred["score"]
                    }
                    st.session_state.feedback_log.append(feedback_entry)
                    st.success(f"✅ Merci! Le modèle a bien classé '{best_pred['class_name']}'")
                    st.balloons()

            with feedback_col2:
                if st.button("👎 C'est Faux", key="wrong_btn", use_container_width=True):
                    st.session_state.show_correction = True

            # Si l'utilisateur dit que c'est faux
            if "show_correction" in st.session_state and st.session_state.show_correction:
                st.warning("Quelle est la bonne classe?")
                correct_class = st.selectbox(
                    "Sélectionner la bonne classe:",
                    sorted(class_names),
                    key="correction_select"
                )

                if st.button("📝 Valider la Correction", key="validate_correction", use_container_width=True):
                    feedback_entry = {
                        "type": "incorrect",
                        "predicted_class": best_pred["class_name"],
                        "correct_class": correct_class,
                        "confidence": best_pred["score"]
                    }
                    st.session_state.feedback_log.append(feedback_entry)
                    st.success(f"✅ Correction enregistrée: '{correct_class}'")
                    st.session_state.show_correction = False
                    st.rerun()

        else:
            st.info("👈 Chargez une image et cliquez sur 'Faire une Prédiction'")

    # Historique des feedbacks
    st.divider()
    st.subheader("📋 Historique des Feedbacks")

    if st.session_state.feedback_log:
        feedback_data = []
        correct_count = 0
        incorrect_count = 0

        for fb in st.session_state.feedback_log:
            if fb["type"] == "correct":
                correct_count += 1
                feedback_data.append({
                    "Type": "✅ Correct",
                    "Prédiction": fb["predicted_class"],
                    "Confiance": f"{fb['confidence']*100:.1f}%"
                })
            else:
                incorrect_count += 1
                feedback_data.append({
                    "Type": "❌ Faux",
                    "Prédiction": fb["predicted_class"],
                    "Réalité": fb["correct_class"],
                    "Confiance": f"{fb['confidence']*100:.1f}%"
                })

        # Statistiques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Feedbacks", len(st.session_state.feedback_log))
        with col2:
            st.metric("Correct ✅", correct_count)
        with col3:
            st.metric("Faux ❌", incorrect_count)

        # Tableau
        st.dataframe(feedback_data, use_container_width=True)

        # Bouton pour sauvegarder
        if st.button("💾 Sauvegarder les Feedbacks", use_container_width=True):
            feedback_file = Path("feedback_log.json")
            with open(feedback_file, "w") as f:
                json.dump(st.session_state.feedback_log, f, indent=2)
            st.success(f"✅ Feedbacks sauvegardés dans {feedback_file}")

        # Bouton pour réinitialiser
        if st.button("🗑️ Réinitialiser", use_container_width=True):
            st.session_state.feedback_log = []
            st.rerun()
    else:
        st.info("Aucun feedback pour le moment. Testez le modèle!")


if __name__ == "__main__":
    main()

