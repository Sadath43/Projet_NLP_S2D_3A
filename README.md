# Projet NLP - RAG sur PDF

Ce projet implémente un système de RAG (Retrieval-Augmented Generation) basé sur des fichiers PDF stockés localement.

Ce guide donne les étapes pour tester le système.

## Prérequis

- Python doit être installé, idéalement la version 3.10.10.

## Installation

1. **Cloner le dépôt** :

   Dans votre dossier de travail, exécutez successivement ces deux commandes :

   ```sh
   git clone https://github.com/Sadath43/Projet_NLP_S2D_3A.git
   cd Projet_NLP_S2D_3A
   ```

2. **Créer et activer un environnement virtuel** :

   Ensuite, passez aux commandes suivantes selon votre système (Windows ou Mac/Linux) :

   ```sh
   python -m venv nlpenv
   source nlpenv/bin/activate  # Sur Mac/Linux
   nlpenv\Scripts\activate  # Sur Windows
   ```

3. **Installer les dépendances** :

   Puis installez les dépendances avec :

   ```sh
   pip install -r requirements.txt
   ```

## Utilisation

1. **Configurer les paramètres** :

   Avant d'exécuter le système, vous pouvez configurer les paramètres (seule la Clé API Hugging Face est **obligatoire** pour que le système démarre) suivants dans le fichier de configuration :

   - `data_dir`: Dossier contenant les fichiers PDF à indexer (par défaut `data/dbase`). `data/dbase` est vide, vous pourrez ajouter des document manuellement.
   - `do_sample`: Active l'échantillonnage pour la génération de texte (`true` par défaut).
   - `embeddings_model`: Modèle d'embeddings utilisé pour la recherche de similarité (`all-MiniLM-L6-v2`).
   - `hf_api_key`: Clé API Hugging Face (**obligatoire** pour utiliser certains modèles).
   - `llm_model`: Modèle de langage utilisé pour la génération (`meta-llama/Llama-3.1-8B-Instruct`).
   - `max_tokens`: Nombre maximal de tokens générés (`500`).
   - `nlp_model`: Modèle NLP utilisé pour le traitement du langage (`en_core_web_trf`).
   - `repetition_penalty`: Pénalité pour éviter les répétitions (`1.1`).
   - `similarity_threshold`: Seuil de similarité pour la récupération des documents (`0.65`).
   - `temperature`: Niveau de variation dans la génération (`0.55`).

2. **Lancer le système** :

    Assurez-vous d'être dans le dossier `Projet_NLP_S2D_3A`, puis exécutez la commande suivante pour lancer l'interface :

    ```sh
    streamlit run cli.py
    ```

    L'interface s'ouvrira automatiquement dans votre navigateur par défaut et vous pourrez interagir avec le système.

---

**Équipe NLP ECC 3A 2024-2025**

