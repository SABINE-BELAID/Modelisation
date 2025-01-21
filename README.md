# **Classification Temporelle de Textes avec Embeddings Statiques**

## **Description**
Ce projet vise à classifier des textes littéraires selon leur période de rédaction en utilisant des modèles d'embeddings statiques tels que **Word2Vec**, **FastText** et **GloVe**. L'objectif principal est d'explorer l'efficacité des embeddings statiques pour la classification temporelle avec des intervalles de 20 ans et 40 ans.

Le projet évalue les performances de ces embeddings en utilisant des modèles de machine learning comme **XGBoost** et **Random Forest**. Des métriques de performance, telles que la précision, le rappel et le F1-score, sont utilisées pour l'évaluation.

---

## **Pipeline**
1. **Chargement des Données** :
   - Les données sont chargées depuis un fichier CSV contenant des textes et leur année de publication.

2. **Prétraitement** :
   - Les textes sont nettoyés et les années de publication sont catégorisées en intervalles (20 ans ou 40 ans).

3. **Extraction des Embeddings** :
   - Les vecteurs d'embeddings sont extraits pour chaque texte en utilisant **Word2Vec**, **FastText**, et **GloVe**.

4. **Classification** :
   - Les embeddings sont utilisés pour entraîner et évaluer des modèles comme **XGBoost** et **Random Forest**.

5. **Évaluation** :
   - Les performances des modèles sont mesurées à l'aide de métriques et des courbes de pertes (log loss) sont générées pour chaque modèle.

6. **Prédiction Année Exacte** :
   - Une régression est effectuée pour prédire l'année exacte de rédaction des textes.

---

## **Résultats**
### **Comparaison des Embeddings par Intervalle**
| Embedding  | Intervalle | Précision Moyenne (%) | Rappel Moyen (%) | F1-Score Moyen (%) | Exactitude (%) |
|------------|------------|------------------------|------------------|---------------------|----------------|
| **Word2Vec** | 20 ans     | 35.38                 | 37.00            | 34.00              | 39.11          |
|             | 40 ans     | 35.38                 | 38.00            | 34.45              | 40.00          |
| **FastText** | 20 ans     | 40.51                 | 42.01            | 39.64              | 45.11          |
|             | 40 ans     | 40.34                 | 45.00            | 40.11              | 40.51          |
| **GloVe**    | 20 ans     | 34.87                 | 38.05            | 34.09              | 39.88          |
|             | 40 ans     | 40.02                 | 36.80            | 33.90              | 34.87          |

### **Prédiction Année Exacte**
| Modèle | MAE (±20 ans) | RMSE  | % Précision dans ±20 ans |
|--------|---------------|-------|--------------------------|
| **Word2Vec** | 24.17         | 32.26 | 67.5%                |
| **FastText** | 22.12         | 30.15 | 69.3%                |
| **GloVe**    | 23.54         | 31.40 | 68.0%                |

### **Visualisation**
Les courbes de log loss (entraînement vs test) et les histogrammes des erreurs absolues sont générés pour une analyse approfondie.

---

## **Prérequis**
1. **Python** >= 3.8
2. **Bibliothèques nécessaires** :
   - `xgboost`, `scikit-learn`, `matplotlib`, `gensim`, `fasttext`, `nltk`, `pandas`, `numpy`

3. **Modèles pré-entraînés** :
   - Word2Vec, FastText, et GloVe doivent être disponibles dans le répertoire `embeddings/`.

---

## **Installation**
1. Clonez le dépôt :
   ```bash
   git clone https://github.com/Modelisation.git
  
