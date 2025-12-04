import streamlit as st
import random
import numpy as np
import pandas as pd
import base64
import json
import requests
import time
from io import BytesIO
from collections import Counter

# --- Configuration et Initialisation ---

action = st.sidebar.selectbox("Choisissez une page :", ["Accueil", "Jeu","classification", "image"])

# --- Contenu de la Page d'Accueil (Markdown) ---
def render_accueil():
    """Affiche le contenu de la page d'accueil."""
    introduction_text = """
# Bienvenue dans le Laboratoire d'Exploration de l'Intelligence Artificielle

Découvrez les fondements de l'IA à travers trois expériences interactives et ludiques. Notre plateforme est conçue pour les débutants curieux qui souhaitent passer de la théorie à la pratique en manipulant le cœur des algorithmes.

Ici, vous n'êtes pas un simple spectateur : vous êtes l'**ingénieur**, le **chercheur** et le **professeur** qui façonne le comportement de l'IA.

---

## 1. 🤖 Le Cerveau de l'IA : Coder la Décision (Raisonnement Séquentiel)

**Prototypes :** Jeu Pierre-Feuille-Ciseaux (PFC-IA)

Comprenez comment une IA prend une décision en se basant sur des règles strictes.

* **Le Concept :** L'**IA symbolique** ou basée sur des règles. Vous écrivez des conditions (`if... else...`) que l'IA exécute pour prédire le prochain coup de l'adversaire.

* **Votre Défi :** Analyser l'historique de jeu, identifier un motif, et programmer la règle de victoire. En modifiant le code, vous voyez immédiatement comment les règles façonnent le comportement intelligent.

* **Ce que vous apprenez :** La logique algorithmique, la programmation conditionnelle et la conception de systèmes experts simples.

## 2. 📊 L'IA qui Trie : Apprendre par l'Exemple (Classification Supervisée)

**Prototypes :** Tri de Cristaux Rares (Classification k-NN)

Découvrez l'**apprentissage automatique** (Machine Learning) en entraînant une IA à trier des objets.

* **Le Concept :** L'IA apprend à reconnaître des catégories en analysant les **caractéristiques numériques** de vos données (la "Taille" et la "Densité" de nos cristaux).

* **Votre Défi :** Créez des ensembles de données d'entraînement clairs (ou ambigus !) et observez comment l'IA utilise la **distance euclidienne** (la règle du "Voisin le Plus Proche") pour classer de nouveaux objets.

* **Ce que vous apprenez :** Le rôle des données d'entraînement, l'espace des caractéristiques, et les principes des algorithmes de classification comme le k-NN.

## 3. 👁️ L'IA qui Voit : Compréhension Visuelle (Modèles Multimodaux)

**Prototypes :** Analyse d'Image avec l'API Gemini

Explorez l'une des technologies d'IA les plus avancées : la **Compréhension Visuelle**.

* **Le Concept :** Les **modèles multimodaux** sont des IA qui peuvent traiter simultanément différents types de données (ici, une image et une question textuelle).

* **Votre Défi :** Téléchargez n'importe quelle image et posez une question pointue. Vous verrez l'IA analyser la scène, identifier les objets, les couleurs, et le contexte pour fournir une réponse pertinente et détaillée.

* **Ce que vous apprenez :** Le fonctionnement des IA génératives et multimodales, l'encodage des images (Base64), et le concept d'une **requête d'API** pour accéder à des services d'IA complexes.

---
### Prêt à commencer votre exploration ?

Choisissez votre premier laboratoire ci-dessous et plongez dans le code et les données !
"""
    st.markdown(introduction_text, unsafe_allow_html=True)

if action == "Accueil":
    render_accueil()

elif action == "Jeu":
    if 'historique_coups_humain' not in st.session_state:
        st.session_state.historique_coups_humain = []
        st.session_state.score_humain = 0
        st.session_state.score_ia = 0
        st.session_state.match_nuls = 0
        st.session_state.dernier_resultat = "En attente du premier coup..."
        st.session_state.explication_ia = "L'IA attend d'apprendre de vos coups."
        st.session_state.dernier_choix_ia = ""

    # Règle personnalisée par défaut pour l'utilisateur
    # NOTE: Le code DOIT inclure 'return' pour que la règle soit appliquée.
    DEFAULT_CUSTOM_RULE = """
        # Règle par Défaut Personnalisée :
        # Si le joueur a joué 'ciseaux' plus de 3 fois, nous prédisons qu'il jouera 'ciseaux' à nouveau.
        if historique.count('ciseaux') >= 3:
            # L'IA prédit 'ciseaux' et joue 'pierre' pour gagner.
            return 'ciseaux', "Règle custom : Je détecte une habitude '✂️', je joue 🪨 pour gagner !"
    """
    if 'custom_rule_code' not in st.session_state:
        st.session_state.custom_rule_code = DEFAULT_CUSTOM_RULE
        
    OPTIONS = ["pierre", "papier", "ciseaux"]
    EMOJIS = {"pierre": "🪨", "papier": "📄", "ciseaux": "✂️"}


    # --- Fonctions de l'IA Modifiable (Cerveau) ---

    def predire_coup(historique, custom_code):
        """
        Fonction de l'IA qui prédit le prochain coup.
        Elle exécute d'abord le code personnalisé, puis les règles par défaut.
        """
        
        # 1. ESSAI DE LA RÈGLE PERSONNALISÉE (Le Défi de l'Étudiant)
        try:
            # IMPORTANT FIX: Nous ajoutons l'indentation de 4 espaces pour chaque ligne
            # pour que le code soit correctement placé DANS la fonction Python générée.
            indented_code = "    " + custom_code.replace('\n', '\n    ')
            
            # Le code de la fonction est construit, incluant un 'return None' de sécurité.
            code_to_exec = f"""
    def custom_prediction(historique):
    {indented_code}
        return None

    result = custom_prediction(historique)
    """
            # Création d'un environnement sûr pour l'exécution du code
            local_vars = {"historique": historique, "len": len, "random": random, "OPTIONS": OPTIONS}
            
            # Exécution du code personnalisé
            exec(code_to_exec, globals(), local_vars)
            
            prediction_result = local_vars.get("result")
            
            # Si la règle personnalisée a fonctionné, elle retourne un tuple valide
            if isinstance(prediction_result, tuple) and len(prediction_result) == 2 and prediction_result[0] in OPTIONS:
                return prediction_result # Règle personnalisée appliquée!
                
        except Exception as e:
            # Affiche l'erreur si le code de l'étudiant est faux
            st.error(f"Erreur dans la Règle Personnalisée : {e}")

        # 2. RÈGLES PAR DÉFAUT (Fallback)
        explication = "J'ai joué de manière aléatoire."
        
        if len(historique) < 3:
            prediction = random.choice(OPTIONS)
            explication = "Historique insuffisant (< 3 coups). J'ai fait un choix totalement aléatoire."
            return prediction, explication
        
        # Détection de motif de répétition des 3 derniers coups
        cp = historique[-3:] 
        if cp.count(cp[0]) == 3:
            prediction = cp[0]
            explication = f"Règle par Défaut : Détection de la séquence répétitive : {cp[0]}, {cp[0]}, {cp[0]}. Je prédis {EMOJIS[prediction]}."
            return prediction, explication
        
        # Cas par défaut : Retour à l'aléatoire si rien n'est trouvé
        prediction = random.choice(OPTIONS)
        explication = "Règle par Défaut : Aucun motif clair n'a été trouvé. Choix aléatoire."
        return prediction, explication


    def trouver_coup_gagnant(choix_predit):
        """Détermine le coup que l'IA doit jouer pour battre la prédiction."""
        if choix_predit == "pierre":
            return "papier"
        elif choix_predit == "papier":
            return "ciseaux"
        else: # ciseaux
            return "pierre"


    # --- Logique du Jeu et Mise à Jour ---

    def determiner_resultat(choix_humain, choix_ia):
        """Détermine qui gagne, met à jour le score et retourne le message."""
        if choix_humain == choix_ia:
            st.session_state.match_nuls += 1
            return "Match nul !"
        elif (choix_humain == "pierre" and choix_ia == "ciseaux") or \
            (choix_humain == "papier" and choix_ia == "pierre") or \
            (choix_humain == "ciseaux" and choix_ia == "papier"):
            st.session_state.score_humain += 1
            return "Victoire ! L'IA s'est fait battre !"
        else:
            st.session_state.score_ia += 1
            return "Défaite... L'IA a gagné ce tour."

    def jouer_un_tour(choix_humain):
        """Fonction principale appelée lors du clic sur le bouton."""
        
        # ÉTAPE A: Prédiction de l'IA (le cerveau)
        prediction_humain, explication = predire_coup(
            st.session_state.historique_coups_humain, 
            st.session_state.custom_rule_code
        )
        
        # ÉTAPE B: Le Coup de l'IA
        choix_ia = trouver_coup_gagnant(prediction_humain)
        
        # ÉTAPE C: Déterminer le résultat
        resultat_message = determiner_resultat(choix_humain, choix_ia)
        
        # ÉTAPE D: Mise à jour de l'historique et de l'état
        st.session_state.historique_coups_humain.append(choix_humain) 
        st.session_state.dernier_resultat = resultat_message
        st.session_state.explication_ia = explication
        st.session_state.dernier_choix_ia = choix_ia


    # --- Interface Utilisateur (Streamlit) ---

    st.set_page_config(page_title="PFC-IA Pédagogique", layout="wide")

    st.markdown("""
    # 🤖 Prototype : Découvrir l'IA par le Code (Pierre-Feuille-Ciseaux)
    **Le défi :** Écrivez une règle de code pour rendre l'IA plus intelligente que l'humain en analysant son historique de jeu !
    """, unsafe_allow_html=True)

    # --- Conteneur principal (Jeu) ---
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Votre Choix")
        # Utilisation du mapping pour l'affichage esthétique des boutons
        for option in OPTIONS:
            if st.button(f"{EMOJIS[option]} {option.capitalize()}", key=option, use_container_width=True):
                # Lancement du tour de jeu au clic
                jouer_un_tour(option)

    with col2:
        st.subheader("Résultat & Score")
        st.markdown(f"**Score :** 🧍 {st.session_state.score_humain} - {st.session_state.score_ia} 🤖")
        st.markdown(f"**Nuls :** {st.session_state.match_nuls}")
        st.markdown("---")
        st.markdown(f"**Dernier Résultat :** **{st.session_state.dernier_resultat}**")
        
        if st.session_state.dernier_choix_ia:
            choix_ia_display = st.session_state.dernier_choix_ia.capitalize()
            st.info(f"L'IA a joué : **{EMOJIS[st.session_state.dernier_choix_ia]} {choix_ia_display}**")
            
        if st.button("Réinitialiser le Jeu et l'Historique", type="secondary"):
            for key in st.session_state.keys():
                if key not in ['custom_rule_code', 'rule_input']: # Conserver la règle saisie
                    del st.session_state[key]
            st.rerun()

    with col3:
        st.subheader("🧠 Cerveau de l'IA")
        st.warning(st.session_state.explication_ia)
        st.markdown("---")
        st.caption("Historique des 10 derniers coups de l'humain :")
        st.code(st.session_state.historique_coups_humain[-10:])

    st.markdown("---")

    # --- Section Modifiable de l'IA (Le Défi Code) ---
    st.subheader("💻 Modifiez le Code de Prédiction de l'IA (Expérimentation)")
    st.markdown("""
    Votre défi est d'écrire une règle en Python qui rend l'IA plus intelligente ! 
    Elle doit analyser la liste `historique` et **retourner un tuple** `(coup_prédit, explication_pour_l'utilisateur)` en cas de succès.

    **Variables disponibles :** `historique` (la liste de tous vos coups).

    **Attention à l'indentation ! Votre code doit commencer dès la première colonne.**
    """)

    # Champ de texte modifiable pour la règle de l'IA
    new_rule_code = st.text_area(
        "Votre Règle Personnalisée (Le code s'exécutera au prochain coup) :",
        value=st.session_state.custom_rule_code,
        height=200,
        key='rule_input' # Clé pour gérer l'entrée
    )

    # Mise à jour de l'état de la règle
    if new_rule_code != st.session_state.custom_rule_code:
        st.session_state.custom_rule_code = new_rule_code
        st.rerun() # Re-lancer pour enregistrer la nouvelle règle

    st.markdown("""
    <div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>
        **Exemple de règle simple qui fonctionne :**
        <pre>
        # Si les 4 derniers coups sont 'papier', je parie que l'utilisateur va changer.
        if len(historique) >= 4 and historique[-4:] == ['papier', 'papier', 'papier', 'papier']:
            return 'pierre', "J'ai utilisé votre règle : après 4 📄, je prédis 🪨."
        </pre>
    </div>
    """, unsafe_allow_html=True)


elif action=="classification":
    # Utilisation de l'état de session Streamlit pour maintenir les données d'entraînement
    if 'historique_cristaux' not in st.session_state:
        # Colonnes : 'Taille' (X), 'Densite' (Y), 'Type' (Bleu/Vert), 'Couleur' (pour le graphique)
        st.session_state.historique_cristaux = pd.DataFrame(columns=['Taille', 'Densite', 'Type', 'Couleur'])
    if 'prediction_resultat' not in st.session_state:
        st.session_state.prediction_resultat = (None, "En attente...")

    # Couleurs pour le graphique et Emojis
    COULEURS = {'Bleu': '#1E90FF', 'Vert': '#3CB371'}
    EMOJIS = {'Bleu': '🟦', 'Vert': '🟩'}
    OPTIONS_TYPE = ['Bleu', 'Vert']

    st.set_page_config(page_title="Classification IA Pédagogique", layout="wide")

    st.markdown("""
    # 🤖 Simulateur de Classification : Apprentissage par Caractéristiques
    Ce prototype montre comment l'IA apprend à classer des objets (cristaux) à partir de leurs **caractéristiques numériques** (Taille et Densité).

    **L'IA utilise la règle simple du "Voisin le Plus Proche" (k-NN).**
    """, unsafe_allow_html=True)

    # --- 1. Logique d'Entraînement (Ajout de Données) ---

    st.header("1. Entraînement de l'IA (Créer les Données)")
    st.caption("Définissez la Taille et la Densité du cristal, puis étiquetez-le (Bleu ou Vert).")

    col_input_1, col_input_2, col_input_3 = st.columns([1, 1, 2])

    with col_input_1:
        taille_input = st.slider("Taille (Caractéristique X)", 0.0, 10.0, 5.0, 0.1)

    with col_input_2:
        densite_input = st.slider("Densité (Caractéristique Y)", 0.0, 10.0, 5.0, 0.1)

    def ajouter_cristal(type_cristal):
        """Ajoute le point de données étiqueté à l'historique."""
        
        # Création du nouveau point
        nouveau_point = {
            'Taille': taille_input,
            'Densite': densite_input,
            'Type': type_cristal,
            'Couleur': COULEURS[type_cristal]
        }
        
        # Ajout au DataFrame via l'état de session
        new_df = pd.DataFrame([nouveau_point])
        st.session_state.historique_cristaux = pd.concat(
            [st.session_state.historique_cristaux, new_df], ignore_index=True
        )
        # Réinitialise le résultat de prédiction après l'ajout d'une nouvelle donnée
        st.session_state.prediction_resultat = (None, "En attente...")


    with col_input_3:
        st.markdown("### Étiquetage")
        col_btn_1, col_btn_2 = st.columns(2)
        with col_btn_1:
            if st.button("Étiqueter comme 🟦 BLEU", use_container_width=True, type="primary"):
                ajouter_cristal('Bleu')
        with col_btn_2:
            if st.button("Étiqueter comme 🟩 VERT", use_container_width=True, type="secondary"):
                ajouter_cristal('Vert')

    # --- 2. Visualisation des Données d'Entraînement ---

    st.header("2. Visualisation des Données (Espace des Caractéristiques)")
    st.caption(f"Nombre de points d'entraînement : {len(st.session_state.historique_cristaux)}")

    if not st.session_state.historique_cristaux.empty:
        # Affichage du nuage de points
        st.scatter_chart(
            st.session_state.historique_cristaux, 
            x='Taille', 
            y='Densite', 
            color='Couleur', 
            height=400
        )
        # 
    else:
        st.warning("Ajoutez des points d'entraînement (Taille, Densité) ci-dessus pour commencer.")

    # --- 3. Logique de Prédiction de l'IA (Le Cerveau) ---

    def calculer_distance_euclidienne(p1, p2):
        """Calcule la distance entre deux points dans l'espace à 2 dimensions."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def predire_type(nouveau_point, historique_df):
        """
        Simule la classification k-NN (avec k=1).
        Trouve le point le plus proche dans l'historique.
        """
        if historique_df.empty:
            return None, "Erreur : L'IA n'a pas de données d'entraînement !"
        
        distances = []
        
        # L'IA parcourt toutes les données d'entraînement
        for index, row in historique_df.iterrows():
            point_entrainement = (row['Taille'], row['Densite'])
            
            # Calcul de la distance
            distance = calculer_distance_euclidienne(nouveau_point, point_entrainement)
            distances.append((distance, row['Type'], point_entrainement))
            
        # Tri des distances pour trouver le plus proche (k=1)
        distances.sort(key=lambda x: x[0])
        
        meilleur_match = distances[0]
        type_predit = meilleur_match[1]
        coordonnees_voisin = meilleur_match[2]
        
        explication = f"""
        Le cristal a été classé comme **{type_predit.upper()}** {EMOJIS[type_predit]}. 
        
        **Raison (Voisin le Plus Proche) :** Le point d'entraînement le plus proche de votre nouveau cristal 
        ({nouveau_point[0]}, {nouveau_point[1]}) est un cristal **{type_predit}** situé à ({coordonnees_voisin[0]}, {coordonnees_voisin[1]}).
        """
        return type_predit, explication


    # --- 4. Section de Test (Prédiction) ---

    st.markdown("---")
    st.header("3. Test de l'IA (Demander une Prédiction)")
    st.caption("Définissez un nouveau cristal que l'IA doit classer.")

    col_test_1, col_test_2, col_test_3 = st.columns([1, 1, 2])

    with col_test_1:
        taille_test = st.slider("Taille du Cristal à Tester (X)", 0.0, 10.0, 5.0, 0.1, key="taille_test")

    with col_test_2:
        densite_test = st.slider("Densité du Cristal à Tester (Y)", 0.0, 10.0, 5.0, 0.1, key="densite_test")

    nouveau_point_a_tester = (taille_test, densite_test)

    def executer_prediction():
        """Exécute la prédiction et met à jour l'état."""
        type_predit, explication_ia = predire_type(nouveau_point_a_tester, st.session_state.historique_cristaux)
        st.session_state.prediction_resultat = (type_predit, explication_ia)

    with col_test_3:
        st.markdown("### Action")
        if st.button("Demander la Prédiction à l'IA", type="primary", use_container_width=True):
            executer_prediction()
        
        if st.button("Réinitialiser toutes les données", type="secondary", use_container_width=True):
            st.session_state.historique_cristaux = pd.DataFrame(columns=['Taille', 'Densite', 'Type', 'Couleur'])
            st.session_state.prediction_resultat = (None, "En attente...")
            st.rerun() # Re-lancer pour un nettoyage complet


    # --- 5. Affichage des Résultats et Explication Pédagogique ---

    st.markdown("---")
    st.subheader("Résultat de la Classification")

    type_predit, explication_ia = st.session_state.prediction_resultat

    if type_predit is not None:
        st.info(f"**Prédiction de l'IA :** Ce cristal est de type **{type_predit.upper()}** {EMOJIS[type_predit]}")
        
        st.markdown("### 🧠 Explication Pédagogique (Le Cerveau de l'IA)")
        st.warning(explication_ia)
        
    else:
        st.info(explication_ia)
        
    st.markdown("""
    ---
    ### 💡 Le Défi pour les Étudiants
    Le k-NN est l'une des IA les plus simples ! 
    **Défi :** Créez des données d'entraînement (les points) de manière à ce que l'IA **se trompe** sur une prédiction. Comment positionner vos points pour que le "Voisin le Plus Proche" soit du mauvais type ?
    """)

elif action=="image":
 
    API_KEY = "AIzaSyCTtTqI5T_QENkqUj46C8D9TOdNP688tDM" # Clé API 
    MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"

    st.set_page_config(page_title="IA de Compréhension Visuelle", layout="wide")

    st.markdown("""
    # 👁️ Compréhension Visuelle : L'IA qui "voit"
    Ce prototype utilise un **Modèle Multimodal** (qui gère texte et images) pour analyser une photo et répondre à vos questions à son sujet.
    """)

    # Description de la technologie utilisée pour la pédagogie
    st.info("""
    ### Le Modèle Multimodal
    Contrairement aux prototypes précédents, cette IA est capable de traiter **simultanément** les données visuelles (l'image) et le texte (votre question). C'est le principe des modèles dits **multimodaux**.
    """)
    # Insertion d'un diagramme pour expliquer le concept de multimodalité
    st.markdown("", unsafe_allow_html=True)


    # --- Fonction de Conversion d'Image (Requis par l'API) ---

    def image_to_base64(image_file):
        """
        Convertit l'objet fichier téléchargé par Streamlit en chaîne Base64
        pour être inclus dans la requête API.
        """
        try:
            # Lire le contenu du fichier
            bytes_data = image_file.read()
            
            # Déterminer le type MIME
            mime_type = image_file.type
            
            # Encoder en Base64
            base64_encoded_data = base64.b64encode(bytes_data).decode('utf-8')
            
            return base64_encoded_data, mime_type
        except Exception as e:
            st.error(f"Erreur lors de la conversion de l'image : {e}")
            return None, None


    # --- Fonction d'Appel à l'API Gemini avec Backoff ---

    def call_gemini_api(prompt, base64_image_data, mime_type):
        """
        Appelle l'API Gemini pour la compréhension d'image avec une logique de
        nouvelles tentatives (exponential backoff).
        """
        
        # Construction de la partie image du contenu
        image_part = {
            "inlineData": {
                "mimeType": mime_type,
                "data": base64_image_data
            }
        }

        # Construction du contenu (prompt et image)
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        image_part
                    ]
                }
            ],
        }

        headers = {'Content-Type': 'application/json'}
        
        # Gestion des tentatives (exponential backoff)
        max_retries = 3
        for i in range(max_retries):
            try:
                # Effectuer l'appel à l'API
                response = requests.post(
                    f"{API_URL}?key={API_KEY}", 
                    headers=headers, 
                    data=json.dumps(payload)
                )
                response.raise_for_status() # Lève une exception pour les codes d'erreur HTTP (4xx ou 5xx)
                
                result = response.json()
                
                # Extraction du texte généré
                candidate = result.get('candidates', [{}])[0]
                generated_text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'Erreur: Contenu non trouvé.')
                
                return generated_text
                
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429 and i < max_retries - 1:
                    # Gérer le cas de limitation de débit (Rate Limit) avec backoff
                    sleep_time = 2 ** i
                    time.sleep(sleep_time)
                    continue
                else:
                    st.error(f"Erreur HTTP lors de l'appel à l'API : {e}")
                    return f"Erreur de l'API: {e}"
            except Exception as e:
                st.error(f"Erreur inattendue : {e}")
                return f"Erreur inattendue: {e}"

        return "Échec de l'appel après plusieurs tentatives."


    # --- Interface Utilisateur Streamlit (Le Code Principal) ---

    col_upload, col_query = st.columns([1, 2])

    with col_upload:
        st.subheader("1. Télécharger l'Image")
        uploaded_file = st.file_uploader(
            "Choisissez une image (JPEG, PNG)", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file:
            # Afficher l'image pour l'utilisateur
            st.image(uploaded_file, caption="Image à analyser", use_column_width=True)

    with col_query:
        st.subheader("2. Posez votre Question à l'IA")
        
        user_prompt = st.text_area(
            "Votre question (Ex: 'Qu'est-ce que cet objet ?' ou 'Décrivez l'arrière-plan')",
            value="Décrivez ce que vous voyez, la couleur principale, et devinez où cette photo a été prise.",
            height=150
        )
        
        if st.button("Lancer l'Analyse IA", type="primary", use_container_width=True):
            
            if uploaded_file is None:
                st.error("Veuillez d'abord télécharger une image pour lancer l'analyse.")
            elif not user_prompt.strip():
                st.error("Veuillez entrer une question pour l'IA.")
            else:
                # Processus d'analyse
                with st.spinner("L'IA est en train d'analyser l'image..."):
                    
                    # Étape 1: Conversion de l'image
                    base64_data, mime_type = image_to_base64(uploaded_file)
                    
                    if base64_data:
                        # Étape 2: Appel à l'API
                        ai_response = call_gemini_api(user_prompt, base64_data, mime_type)
                        
                        # Stocker la réponse dans l'état de session pour l'afficher après
                        st.session_state.ai_analysis_result = ai_response
                        st.rerun() # Re-lancer pour afficher le résultat

    # --- 3. Affichage du Résultat ---

    st.markdown("---")
    st.subheader("3. Réponse de l'IA")

    if 'ai_analysis_result' in st.session_state:
        st.success("Analyse Complète !")
        st.markdown(st.session_state.ai_analysis_result)
    else:
        st.info("La réponse de l'IA s'affichera ici après l'analyse.")

    st.markdown("""
    ---
    ### 💡 Le Défi pour les Étudiants
    **Défi :** Essayez de tromper l'IA ! Téléchargez une image ambiguë (par exemple, une photo floue ou une illusion d'optique) et voyez si l'IA parvient à la décrire correctement.
    """)