"""
=============================================================================
 DIAGNOSTIC DES FAUX POSITIFS
=============================================================================
 But : comprendre POURQUOI le modèle classe ~320 fenêtres bénignes comme
 attaques. On entraîne le modèle sur une partie des données, on repère les
 fenêtres bénignes mal classées (faux positifs), et on compare leurs
 caractéristiques MOYENNES à celles :
   - des vraies attaques,
   - du bénin correctement classé.

 Si les faux positifs ressemblent aux attaques (petits paquets, intervalles
 courts) plutôt qu'au bénin bien classé, l'hypothèse "les 4 caractéristiques
 ne suffisent pas à distinguer un flood d'un trafic bénin dense" est confirmée.

 Réutilise les fonctions de chargement de cnn_ddos_crossval.py.
=============================================================================
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from cnn_ddos_crossval import (
    charger_dossiers, normaliser, construire_modele,
    DOSSIERS_PCAP, NB_CARACTERISTIQUES, GRAINE,
    TAILLE, INTERVALLE, PROTOCOLE, PORT,
)


def profil(fenetres):
    """Caractéristiques moyennes d'un ensemble de fenêtres (moyenne sur tous
    les paquets de toutes les fenêtres), pour chacune des 4 caractéristiques."""
    plat = fenetres.reshape(-1, NB_CARACTERISTIQUES)
    return plat.mean(axis=0)


def main():
    tf.keras.utils.set_random_seed(GRAINE)

    print("=== Chargement ===")
    fenetres, etiquettes = charger_dossiers(DOSSIERS_PCAP)

    # un simple découpage train/test (stratifié) suffit pour le diagnostic
    idx = np.arange(len(fenetres))
    idx_tr, idx_te = train_test_split(idx, test_size=0.25,
                                      stratify=etiquettes, random_state=GRAINE)
    fen_tr, fen_te = fenetres[idx_tr], fenetres[idx_te]
    etq_tr, etq_te = etiquettes[idx_tr], etiquettes[idx_te]

    # IMPORTANT : on garde une copie NON normalisée du test pour lire les
    # caractéristiques en unités réelles (octets, secondes), pas standardisées.
    fen_te_brut = fen_te.copy()

    fen_tr_n, fen_te_n = normaliser(fen_tr, fen_te)
    modele = construire_modele()
    modele.fit(fen_tr_n, etq_tr, epochs=15, batch_size=64, verbose=0)

    proba = modele.predict(fen_te_n, verbose=0).ravel()
    pred  = (proba >= 0.5).astype(int)
    vrai  = etq_te.astype(int)

    # masques sur le jeu de test (indices alignés avec fen_te_brut)
    faux_positifs = (pred == 1) & (vrai == 0)   # bénin classé attaque
    benin_ok      = (pred == 0) & (vrai == 0)   # bénin bien classé
    vraie_attaque = (vrai == 1)

    print(f"\nTest : {len(vrai)} fenêtres")
    print(f"  faux positifs (bénin -> attaque) : {int(faux_positifs.sum())}")
    print(f"  bénin bien classé                : {int(benin_ok.sum())}")
    print(f"  vraies attaques                  : {int(vraie_attaque.sum())}")

    noms = ["taille (octets)", "intervalle (s) ", "protocole      ", "port           "]
    print("\n" + "=" * 70)
    print(f"{'Caractéristique moyenne':24} {'FAUX POSITIFS':>14} {'bénin OK':>12} {'attaques':>12}")
    print("=" * 70)
    p_fp = profil(fen_te_brut[faux_positifs]) if faux_positifs.any() else np.full(NB_CARACTERISTIQUES, np.nan)
    p_ok = profil(fen_te_brut[benin_ok])      if benin_ok.any()      else np.full(NB_CARACTERISTIQUES, np.nan)
    p_at = profil(fen_te_brut[vraie_attaque]) if vraie_attaque.any() else np.full(NB_CARACTERISTIQUES, np.nan)
    for i, nom in enumerate(noms):
        print(f"{nom:24} {p_fp[i]:>14.4f} {p_ok[i]:>12.4f} {p_at[i]:>12.4f}")
    print("=" * 70)
    print("\nLecture : si la colonne FAUX POSITIFS ressemble à ATTAQUES (petite")
    print("taille, intervalle court) plutôt qu'à BÉNIN OK, alors les 4")
    print("caractéristiques ne suffisent pas à séparer ces fenêtres.")


if __name__ == "__main__":
    main()
