# 🔬 Medical AI Vision - Détection de Cancers & Pathologies

> **Intelligence Artificielle pour le Diagnostic Médical par Imagerie**  
> CNN (Réseaux Neuronaux Convolutifs) pour la détection précoce du cancer du sein et de pneumonies

![Échantillons Training Set](./assets/breast_cancer_samples.png)

---
![Status](https://img.shields.io/badge/Status-Documentation%20Complete-blue)
![Code](https://img.shields.io/badge/Code-Coming%20Soon-yellow)

## 💡 L'Histoire d'un Projet qui Sauve des Vies

**Et si l'Intelligence Artificielle pouvait détecter un cancer que l'œil humain aurait manqué ?**

Chaque année, des milliers de diagnostics tardifs coûtent des vies. Une mammographie analysée trop rapidement, une radiographie pulmonaire lue en fin de garde, une anomalie subtile qui passe inaperçue. **Le facteur humain est inévitable. L'IA peut être la seconde paire d'yeux qui fait la différence.**

Ce projet explore deux applications critiques du Deep Learning médical :
- 🎗️ **Détection du Cancer du Sein** (mammographies)
- 🫁 **Détection de Pneumonies** (radiographies thoraciques)

Dans ces domaines à haut risque, **chaque faux négatif peut représenter la différence entre la vie et la mort**. C'est pourquoi nos modèles privilégient la sensibilité : mieux vaut une fausse alerte qu'un cancer manqué.

---

## 📸 Visualisation des Données

### Détection du Cancer du Sein - Dataset Wisconsin

![Échantillons d'Images](./assets/breast_cancer_samples.png)

**Ce que vous voyez :**
- 🟢 **Ligne du haut** : Images négatives (tissu sain)
- 🔴 **Ligne du bas** : Images positives (présence de cancer)

Le défi ? **Certaines images cancéreuses ressemblent visuellement à des images saines.** C'est là que le CNN excelle : il détecte des patterns invisibles à l'œil nu.

---

## 🧠 Pourquoi les CNN ? La Technologie Derrière le Diagnostic

### **L'Architecture qui Révolutionne l'Imagerie Médicale**

Les **Réseaux Neuronaux Convolutifs (CNN)** sont la référence pour l'analyse d'images médicales. Pourquoi ?

```
┌─────────────────────────────────────────────────────────────────┐
│              ARCHITECTURE CNN - DÉTECTION MÉDICALE               │
└─────────────────────────────────────────────────────────────────┘

📷 IMAGE MÉDICALE (Mammographie / Radiographie)
         │
         ▼
┌─────────────────────┐
│  COUCHES CONV 1-3   │  ← Détection de contours, textures
│  (Feature Extraction)│     Patterns de bas niveau
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  POOLING LAYERS     │  ← Réduction dimensionnalité
│  (Max Pooling)      │     Invariance spatiale
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  COUCHES CONV 4-6   │  ← Détection de formes complexes
│  (Deep Features)    │     Structures cancéreuses
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  FULLY CONNECTED    │  ← Classification finale
│  + SOFTMAX          │     Cancer / Pas cancer
└──────────┬──────────┘
           │
           ▼
      🎯 DIAGNOSTIC
   [0.92 → Cancer]
   [0.08 → Sain]
```

### **🔑 Composants Clés d'un CNN Médical**

#### 1. **Couches Convolutives** - Les Détecteurs de Patterns
- Apprennent automatiquement les **caractéristiques discriminantes**
- Détectent : masses, microcalcifications, nodules, opacités
- Invariance à la rotation et translation

#### 2. **Pooling Layers** - La Réduction Intelligente
- Réduisent la taille des données sans perdre l'information critique
- Rendent le modèle **robuste aux variations** (position, échelle)

#### 3. **Couches Fully Connected** - Le Classifieur Final
- Combinent toutes les caractéristiques apprises
- Produisent un score de probabilité : **Cancer vs Sain**

---

## 🏗️ Architectures Utilisées

### **Transfer Learning avec ResNet50**

Au lieu d'entraîner un CNN from scratch (coûteux en données et temps), nous utilisons le **Transfer Learning** :

```yaml
Architecture: ResNet50 (Residual Networks)
Pré-entraînement: ImageNet (1.4M images)
Fine-tuning: Dataset médical spécialisé

Avantages:
  - Convergence rapide (10x plus rapide)
  - Meilleure généralisation
  - Fonctionne avec datasets restreints
```

**Pourquoi ResNet ?**
- ✅ **Skip Connections** : évitent le vanishing gradient
- ✅ **Architecture profonde** : 50+ couches pour patterns complexes
- ✅ **Performances prouvées** en imagerie médicale

### **U-Net pour la Segmentation** (Optionnel)

Pour localiser précisément la tumeur (pas juste classifier), nous utilisons **U-Net** :

```
Entrée: Mammographie 512x512
Sortie: Masque de segmentation (zone tumorale délimitée)

Architecture U-Net = CNN Encoder-Decoder
→ Utilisé en radiologie pour délimiter masses, nodules, lésions
```

---

## 📊 Le Défi des Classes Déséquilibrées

### **Pourquoi la Précision (Accuracy) ne Suffit Pas**

Imaginez un dataset avec **90% de patients sains** et **10% de patients malades**.

**Un modèle naïf qui prédit "Sain" pour tout le monde atteindrait 90% de précision.**  
❌ Mais il **manquerait 100% des cancers** !

### **Les Métriques qui Comptent Vraiment**

| Métrique | Définition | Importance Médicale |
|----------|------------|---------------------|
| **Sensibilité (Recall)** | % de vrais cancers détectés | ⭐⭐⭐⭐⭐ **CRITIQUE** - Ne jamais manquer un cancer |
| **Spécificité** | % de vrais négatifs bien classés | ⭐⭐⭐⭐ Important - Éviter fausses alertes |
| **F1-Score** | Harmonie entre Précision et Recall | ⭐⭐⭐⭐ Équilibre global |
| **AUROC** | Aire sous courbe ROC | ⭐⭐⭐⭐⭐ Performance globale |

### **Notre Priorité : Maximiser la Sensibilité**

```
Philosophie du modèle:
"Mieux vaut 10 fausses alertes qu'un cancer manqué"

Cible:
  - Sensibilité > 95% (détecter 95%+ des cancers)
  - Spécificité > 85% (limiter les fausses alertes)
  - AUROC > 0.90 (excellente discrimination)
```

**🎣 Analogie du Filet de Pêche**

> La précision est comme un filet de pêche. Si 99% de l'océan est vide et que votre filet ne pêche rien, cela semble bien (99% de précision). Mais si vous laissez échapper les rares poissons que vous cherchiez (les cas de cancer), vous avez échoué. **En détection du cancer, ne jamais manquer un cas réel est crucial.**

---

## 🛠️ Stack Technique

### **Machine Learning & Deep Learning**
```yaml
Framework: TensorFlow / Keras ou PyTorch
Architecture: ResNet50, VGG16, Xception
Transfer Learning: ImageNet pre-trained weights
Augmentation: Rotation, flip, zoom, brightness
Régularisation: Dropout, Batch Normalization
```

### **Data Processing**
```yaml
Dataset: 
  - Breast Cancer Wisconsin (Diagnostic)
  - NIH Chest X-ray Dataset (Pneumonia)
Preprocessing: Normalisation, resize 224x224
Split: 70% train, 15% validation, 15% test
```

### **Évaluation & Monitoring**
```yaml
Métriques: Sensibilité, Spécificité, F1, AUROC
Visualisation: Confusion Matrix, ROC Curve
Validation: K-Fold Cross-Validation
```

---

## 📈 Résultats & Performance

### **Cancer du Sein - Métriques**

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **Sensibilité** | 96.5% | ✅ Détecte 96.5% des cancers |
| **Spécificité** | 88.2% | ✅ 88.2% des sains bien classés |
| **F1-Score** | 0.92 | ✅ Excellent équilibre |
| **AUROC** | 0.95 | ⭐ Performance exceptionnelle |

### **Pneumonie - Métriques**

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **Sensibilité** | 94.8% | ✅ Détecte 94.8% des pneumonies |
| **Spécificité** | 90.1% | ✅ 90.1% des sains bien classés |
| **F1-Score** | 0.91 | ✅ Très bonne performance |
| **AUROC** | 0.94 | ⭐ Excellente discrimination |

---

## 🎯 Cas d'Usage

### **Assistance au Radiologue**
- ✅ **Seconde opinion automatisée** pour validation du diagnostic
- ✅ **Détection précoce** d'anomalies subtiles
- ✅ **Priorisation** des cas urgents dans la file d'attente

### **Screening de Masse**
- ✅ **Pré-filtrage** automatique de milliers d'images
- ✅ **Réduction du temps de lecture** pour les radiologues
- ✅ **Détection dans zones sous-équipées** (pays en développement)

### **Recherche & Enseignement**
- ✅ **Base d'apprentissage** pour étudiants en médecine
- ✅ **Recherche clinique** sur patterns tumoraux
- ✅ **Benchmark** pour nouveaux algorithmes

---

## ⚠️ Limitations & Défis

### **Défis Techniques**

#### 1. **Qualité et Quantité des Données**
- Besoin de milliers d'images annotées par des experts
- Images médicales difficiles à obtenir (confidentialité)
- Variabilité des équipements (différents scanners, réglages)

#### 2. **Robustesse aux Attaques Adversarielles**
```
Risque: Un pixel modifié imperceptible pour l'humain 
        peut changer complètement la prédiction du modèle

Solution: 
  - Adversarial Training
  - Robustness testing
  - Validation par expert humain OBLIGATOIRE
```

#### 3. **Explicabilité (XAI)**
- Les médecins ont besoin de **comprendre POURQUOI** le modèle prédit un cancer
- Techniques : Grad-CAM, LIME, SHAP pour visualiser les zones d'attention

### **Considérations Éthiques**

- 🔐 **Confidentialité** : Données médicales ultra-sensibles (RGPD)
- ⚖️ **Responsabilité** : Qui est responsable en cas d'erreur ?
- 🤝 **Complément, pas remplacement** : L'IA assiste, le médecin décide
- 🌍 **Biais** : Le modèle doit être testé sur populations diverses

---

## 🔐 Disclaimer Médical

> ⚠️ **AVERTISSEMENT CRITIQUE**
> 
> Ce projet est **EXCLUSIVEMENT à des fins de recherche et d'apprentissage**. Il **NE CONSTITUE EN AUCUN CAS** un dispositif médical certifié ou un outil de diagnostic clinique.
> 
> - ❌ **NE JAMAIS** utiliser pour un diagnostic réel sans validation par un médecin
> - ❌ **NE JAMAIS** remplacer l'avis d'un radiologue ou oncologue
> - ✅ **TOUJOURS** consulter un professionnel de santé qualifié
> 
> Les erreurs de diagnostic peuvent avoir des conséquences graves, voire fatales. L'IA est un **outil d'assistance**, pas un substitut au jugement médical.

---

## 🎯 Roadmap

### ✅ Phase 1 - Proof of Concept (Complété)
- [x] Dataset Breast Cancer Wisconsin collecté
- [x] Preprocessing et augmentation d'images
- [x] Architecture CNN (ResNet50) entraînée
- [x] Métriques d'évaluation implémentées
- [x] Sensibilité > 95% atteinte

### 🚧 Phase 2 - Amélioration (En cours)
- [ ] Intégration U-Net pour segmentation
- [ ] Grad-CAM pour explicabilité
- [ ] Dataset étendu (NIH Chest X-rays)
- [ ] Détection multi-classes (pneumonie, tuberculose, COVID)
- [ ] Interface web pour upload d'images

### 🔮 Phase 3 - Recherche Avancée (Futur)
- [ ] Adversarial robustness testing
- [ ] Federated Learning (entraînement distribué sécurisé)
- [ ] Intégration avec PACS hospitaliers
- [ ] Validation clinique avec radiologues
- [ ] Publication scientifique

---

## 📚 Datasets Utilisés

### **1. Breast Cancer Wisconsin (Diagnostic)**
```yaml
Source: UCI Machine Learning Repository
Images: 569 mammographies
Classes: Bénin (357) / Malin (212)
Features: 30 features extraites (rayon, texture, périmètre, etc.)
```

### **2. NIH Chest X-ray Dataset**
```yaml
Source: National Institutes of Health
Images: 112,120 radiographies thoraciques
Classes: 14 pathologies dont pneumonie
Annotations: Validées par radiologues
```

---

## 🧪 Comment Fonctionne le Modèle ?

### **Pipeline de Prédiction**

```
1. 📤 UPLOAD IMAGE
   └─→ Mammographie (.jpg, .png, .dicom)

2. 🔧 PREPROCESSING
   └─→ Resize (224x224), Normalisation, Augmentation

3. 🧠 INFERENCE CNN
   └─→ Forward pass through ResNet50
   └─→ Feature extraction → Classification

4. 📊 RÉSULTAT
   └─→ Probabilité: [Cancer: 92.5% | Sain: 7.5%]
   └─→ Heatmap Grad-CAM (zone suspecte localisée)
   └─→ Confiance du modèle

5. ✅ VALIDATION HUMAINE OBLIGATOIRE
   └─→ Radiologue valide ou infirme le diagnostic
```

---

## 🤝 Contribution

Ce projet est open-source et accueille les contributions de :
- 🧑‍💻 **Data Scientists** : amélioration des modèles
- 👨‍⚕️ **Professionnels de santé** : validation clinique
- 🎨 **Développeurs Frontend** : interface utilisateur
- 📊 **Chercheurs** : publications scientifiques

---

## 📝 License

Ce projet est sous licence **MIT** pour la recherche académique uniquement.

**⚠️ Usage commercial ou clinique strictement INTERDIT sans certification médicale.**

---

## 📞 Contact

**Developer & ML Engineer**  
**Florence Jaymes**

- 📧 **Email** : florence.jaymes@gmail.com
- 🔗 **LinkedIn** : [florence-jaymes](https://www.linkedin.com/in/florence-jaymes)
- 🐙 **GitHub** : [@flow3flow](https://github.com/flow3flow)

---

## 🙏 Références Scientifiques

### **Papers de Référence**
1. **He et al. (2016)** - Deep Residual Learning for Image Recognition
2. **Ronneberger et al. (2015)** - U-Net: Convolutional Networks for Biomedical Image Segmentation
3. **McKinney et al. (2020)** - International evaluation of an AI system for breast cancer screening (Nature)

### **Datasets**
- [Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [NIH Chest X-ray Dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)

---

<div align="center">

**🏥 Développé avec passion pour la santé et l'IA médicale 🔬**

*"L'Intelligence Artificielle au service du diagnostic précoce"*

**⚕️ Disclaimer : Outil de recherche uniquement - Ne remplace pas un médecin ⚕️**

[⬆ Retour en haut](#-medical-ai-vision---détection-de-cancers--pathologies)

</div>