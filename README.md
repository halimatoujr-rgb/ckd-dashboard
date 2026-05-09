# CKD Analytics Dashboard 🩺

Dashboard interactif pour l'analyse du dataset **Chronic Kidney Disease** (UCI ML Repository, ID=336).

Construit avec **Plotly Dash** · Thème sombre · 100% interactif

---

## 📊 Fonctionnalités

- **KPI Cards** — Total patients, CKD, Non-CKD, Âge moyen
- **Scatter Plot interactif** — Choisir les variables X et Y dynamiquement + trendline
- **Donut Chart** — Répartition CKD vs Non-CKD
- **Heatmap** — Matrice de corrélation des variables clés
- **Boxplot dynamique** — Distribution de n'importe quelle variable numérique par diagnostic
- **Graphique métier** — Effet combiné Diabète × Hypertension sur la CKD

---

## 🚀 Installation & Lancement

```bash
# 1. Cloner le repo
git clone https://github.com/TON_USERNAME/ckd-dashboard.git
cd ckd-dashboard

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer le dashboard
python app.py
```

Ouvrir ensuite : [http://127.0.0.1:8050](http://127.0.0.1:8050)

---

## 🗂️ Structure

```
ckd-dashboard/
├── app.py            # Application principale
├── requirements.txt  # Dépendances Python
└── README.md         # Documentation
```

---

## 📦 Dataset

- **Source** : UCI Machine Learning Repository
- **ID** : 336
- **Patients** : 400
- **Variables** : 24 features + 1 target (ckd / notckd)

---

## 🛠️ Stack technique

| Outil | Rôle |
|-------|------|
| Plotly Dash | Framework dashboard |
| Plotly Express | Visualisations |
| Pandas | Traitement des données |
| ucimlrepo | Chargement du dataset |
| statsmodels | Trendline OLS |

---

*Projet réalisé dans le cadre de l'examen Data Visualisation — Licence 2 Big Data*
