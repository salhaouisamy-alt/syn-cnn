# -*- coding: utf-8 -*-
"""
cnn_ddos_tailles.py
Compare plusieurs TAILLES du CNN 1D, tout le reste etant IDENTIQUE :
memes donnees, meme etiquetage (seuil 6), meme validation croisee a 5 plis,
meme normalisation par pli, memes parametres d'entrainement.
Seule la taille du reseau change (nombre de filtres de convolution et
d'unites de la couche dense).

Le script reutilise directement les fonctions de cnn_ddos_crossval.py
(chargement, normalisation, metriques) afin de garantir un protocole
strictement identique a celui du memoire.

A lancer dans le meme dossier que cnn_ddos_crossval.py :
    python cnn_ddos_tailles.py

La sortie (tableau comparatif) est concue pour etre capturee et inseree
dans le memoire.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedKFold

# Reutilisation directe de la chaine existante -> protocole identique.
from cnn_ddos_crossval import (
    charger_dossiers, normaliser, metriques,
    DOSSIERS_PCAP, SEUIL_ETIQUETTE, NB_PLIS, GRAINE,
    NB_PAQUETS, NB_CARACTERISTIQUES,
)

# ---------------------------------------------------------------------------
# Tailles a comparer : (nom, filtres_conv1, filtres_conv2, unites_dense)
# La premiere ligne est le modele actuel du memoire.
# ---------------------------------------------------------------------------
CONFIGURATIONS = [
    ("Compact (actuel)", 16, 32, 16),
    ("Intermediaire", 32, 64, 32),
    ("Plus grand", 64, 128, 64),
]


def construire_modele(f1, f2, dense):
    """Meme architecture que le memoire ; seule la taille varie."""
    modele = models.Sequential([
        layers.Input(shape=(NB_PAQUETS, NB_CARACTERISTIQUES)),
        layers.Conv1D(f1, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(f2, kernel_size=3, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(dense, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    modele.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return modele


def evaluer_taille(fenetres, etiquettes, f1, f2, dense):
    """Validation croisee a NB_PLIS plis pour une taille donnee.
    Retourne (nb_parametres, moyennes, ecarts_types) sur
    (exactitude, precision, rappel, F1)."""
    decoupage = StratifiedKFold(n_splits=NB_PLIS, shuffle=True, random_state=GRAINE)
    scores = []
    n_params = None
    for idx_tr, idx_te in decoupage.split(fenetres, etiquettes):
        fen_tr, fen_te = fenetres[idx_tr], fenetres[idx_te]
        etq_tr, etq_te = etiquettes[idx_tr], etiquettes[idx_te]
        fen_tr, fen_te = normaliser(fen_tr, fen_te)          # recalcul par pli
        modele = construire_modele(f1, f2, dense)
        if n_params is None:
            n_params = modele.count_params()
        modele.fit(fen_tr, etq_tr, epochs=15, batch_size=64, verbose=0)
        exa, pre, rap, f1s, fp, fn, _, _ = metriques(modele, fen_te, etq_te)
        scores.append((exa, pre, rap, f1s))
    scores = np.array(scores)
    return n_params, scores.mean(axis=0), scores.std(axis=0)


def main():
    tf.keras.utils.set_random_seed(GRAINE)

    print("=== Chargement (identique a cnn_ddos_crossval) ===")
    fenetres, etiquettes = charger_dossiers(DOSSIERS_PCAP)
    print(f"Total : {len(fenetres)} fenetres  "
          f"(seuil d'etiquetage = {SEUIL_ETIQUETTE}, {NB_PLIS} plis)\n")

    resultats = []
    for nom, f1, f2, dense in CONFIGURATIONS:
        print(f"--- {nom}  (Conv {f1}->{f2}, Dense {dense}) ---")
        n_params, moy, ect = evaluer_taille(fenetres, etiquettes, f1, f2, dense)
        resultats.append((nom, n_params, moy, ect))
        print(f"    parametres = {n_params}  (~{n_params/1024:.1f} Ko en int8)")
        print(f"    precision = {moy[1]:.3f} +/- {ect[1]:.3f}   "
              f"rappel = {moy[2]:.3f} +/- {ect[2]:.3f}   "
              f"F1 = {moy[3]:.3f} +/- {ect[3]:.3f}\n")

    # ----------------------------- tableau final -----------------------------
    print("=" * 96)
    print(f"{'Configuration':<20}{'Parametres':>11}{'int8':>9}"
          f"{'Precision':>16}{'Rappel':>16}{'F1':>16}")
    print("=" * 96)
    for nom, n, moy, ect in resultats:
        ko = f"{n/1024:.1f} Ko"
        pre = f"{moy[1]:.3f}+/-{ect[1]:.3f}"
        rap = f"{moy[2]:.3f}+/-{ect[2]:.3f}"
        f1s = f"{moy[3]:.3f}+/-{ect[3]:.3f}"
        print(f"{nom:<20}{n:>11}{ko:>9}{pre:>16}{rap:>16}{f1s:>16}")
    print("=" * 96)
    print("\nToutes les configurations partagent exactement les memes donnees,")
    print("le meme etiquetage (seuil 6) et la meme validation croisee a 5 plis.")


if __name__ == "__main__":
    main()
