# Projet de Bachelor: SYN-CNN — Détection d'attaques DDoS par inondation SYN adaptée aux FPGA

Détecteur léger d'attaques DDoS de type **inondation SYN**, conçu pour être embarqué
sur un **FPGA**. Le trafic réseau est représenté sous forme de fenêtres de paquets et
classé par un réseau de neurones convolutif (CNN) 1D volontairement compact
(**2 321 paramètres, ~2,3 Ko** en entiers 8 bits), désigné **SYN-CNN**.

Ce dépôt contient le code source du travail de Bachelor associé.

## Contexte

- **Auteur :** Samy Salhaoui
- **Formation :** Bachelor en Informatique et systèmes de communication (ISC), orientation Sécurité
- **École :** HEPIA — HES-SO Genève
- **Encadrant :** Eryk Schiller
- **Année :** 2026

## Objectif

Concevoir un modèle d'apprentissage automatique capable de détecter les inondations SYN
en temps réel, suffisamment compact pour tenir dans la mémoire limitée d'un FPGA, tout en
minimisant les faux positifs (bloquer un utilisateur légitime étant particulièrement pénalisant).

## Approche

- Le trafic est découpé en **fenêtres de 24 paquets**, chaque paquet décrit par
  **4 caractéristiques d'en-tête** : taille, intervalle inter-paquets, protocole, port de destination.
- Chaque fenêtre (matrice 24 × 4) est classée **bénin / attaque** par un **CNN 1D** :
  `Conv1D → MaxPool → Conv1D → GlobalAveragePooling → Dense → sigmoïde`.
- La conception a évolué en trois politiques d'étiquetage successives, la version retenue
  étiquetant une fenêtre comme « attaque » dès **6 paquets d'attaque sur 24**.
- Évaluation par **validation croisée stratifiée à 5 plis**.

## Structure du dépôt

```
src/
├── cnn_ddos_crossval.py        # Entraînement + évaluation en validation croisée 5 plis (matrices de confusion)
├── cnn_ddos_seuil.py           # Balayage du seuil d'étiquetage des fenêtres (comparaison des performances)
├── cnn_ddos_tailles.py         # Comparaison de plusieurs tailles de modèle (compact / intermédiaire / grand)
├── cnn_ddos_generalisation.py  # Première conception : évaluation par découpage par capture (test de généralisation)
├── diagnostic_faux_positifs.py # Analyse des caractéristiques moyennes des faux positifs
├── diagnostic_a_cheval.py      # Comptage des paquets d'attaque dans les fenêtres « à cheval »
└── diagnostic_syn_sources.py   # Diagnostic des sources de trafic SYN
```

## Prérequis

- Python 3.10+
- Bibliothèques principales : `scapy`, `numpy`, `tensorflow`, `scikit-learn`

## Installation

```bash
# Cloner le dépôt
git clone https://githepia.hesge.ch/samy.salhaoui1/projet_de_bachelor.git
cd projet_de_bachelor

# Créer et activer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# Installer les dépendances
pip install scapy numpy tensorflow scikit-learn
```

## Utilisation

Les scripts lisent des captures réseau au format **`.pcap`** (via `scapy.PcapReader`).
Placez vos captures dans le dossier attendu par les scripts (voir le chemin défini en tête
de chaque fichier) puis exécutez, par exemple :

```bash
# Entraînement et évaluation du modèle retenu (validation croisée)
python src/cnn_ddos_crossval.py

# Étude de l'effet du seuil d'étiquetage
python src/cnn_ddos_seuil.py

# Comparaison des tailles de modèle
python src/cnn_ddos_tailles.py
```

> **Note :** les captures `.pcap` utilisées (banc d'essai HEPIA) ne sont pas incluses
> dans ce dépôt en raison de leur volume. Adaptez les chemins d'accès aux données en tête
> des scripts selon votre environnement.

## Résultats

Avec la configuration retenue (seuil d'étiquetage à 6 paquets), le modèle atteint, en
validation croisée à 5 plis :

| Métrique   | Valeur |
|------------|--------|
| Précision  | 0,987  |
| Rappel     | 0,998  |
| F1-score   | 0,993  |
| Empreinte  | ~2,3 Ko (int8) |

Une comparaison de tailles montre qu'agrandir le réseau (jusqu'à ~15×) n'améliore pas les
performances, ce qui confirme la pertinence du format compact.

## Avertissement

Projet réalisé dans un cadre académique et évalué sur un banc d'essai dédié (captures
d'inondation SYN issues de Mirai). Les performances rapportées valent pour ce périmètre
et ne préjugent pas de la robustesse face à d'autres familles d'attaques DDoS.