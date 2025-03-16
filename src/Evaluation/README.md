# README pour l'évaluation avec Ragas

## Prérequis

Avant d'exécuter le script, assurez-vous d'avoir installé les bibliothèques nécessaires. 

pip install pandas datasets ragas python-dotenv


## Configuration : 

1. Clé API OpenAI :
   - Créez un fichier `.env` dans le même répertoire que `evaluator.py`.
   - Ajoutez votre clé API OpenAI dans ce fichier au format suivant :

   OPENAI_API_KEY=votre_cle_openai

2. Fichier de données :
   - Assurez-vous d'avoir le fichier nommé `ragas_dataset_eval.json` dans le même répertoire. Ce fichier doit contenir vos données d'évaluation au format JSON.

## Exécution du script:

Pour exécuter le script `evaluator.py`, utilisez la commande suivante :

python evaluator.py


Ce que fait le script:
- Charge les variables d'environnement (la clé openai) depuis le fichier `.env`.
- Vérifie l'existence du fichier `ragas_dataset_eval.json`.
- Charge les données JSON et les convertit au format attendu par Ragas.
- Crée un dataset Hugging Face à partir des données.
- Configure les métriques pour l'évaluation : fidélité, pertinence des réponses, et rappel du contexte.
- Évalue le dataset et affiche les résultats.
- Enregistre les résultats dans un fichier CSV nommé `ragas_evaluation_results.csv`.

