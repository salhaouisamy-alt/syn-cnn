"""
=============================================================================
 CNN 1D - Test de GÉNÉRALISATION
 On sépare PAR FICHIER : certaines captures pour entraîner, d'autres (jamais
 vues) pour tester. Objectif : mesurer si le modèle généralise, au lieu de
 réapprendre les mêmes captures.
=============================================================================

 Limite : une seule capture d'attaque (mirai-syn) -> on la coupe dans le temps
 (début = entraînement, fin = test). Côté normal, on tient des captures
 entières hors de l'entraînement (dont RTSP, jamais vu à l'entraînement).

 Lancement :
     python cnn_ddos_generalisation.py
=============================================================================
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scapy.all import IP, TCP, UDP, PcapReader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# --- Configuration ---
GRAINE = 42
NB_PAQUETS = 24
NB_CARACTERISTIQUES = 4
TAILLE, INTERVALLE, PROTOCOLE, PORT = 0, 1, 2, 3
MAX_FENETRES_PAR_FICHIER = 2000

DOSSIER_PCAP = r"C:\Users\jojo_\Desktop\data_pcap_bachelor"

# Captures NORMALES utilisées pour ENTRAÎNER (le modèle les voit)
CAPTURES_TRAIN_NORMAL = [
    "capture_20250908_600sec_0000_http.pcap",
    "capture_20250908_600sec_0000_http+mqtt.pcap",
    "capture_20250908_600sec_0000_mqtt.pcap",
]
# Captures NORMALES gardées pour le TEST (jamais vues -> RTSP inclus)
CAPTURES_TEST_NORMAL = [
    "capture_20250908_600sec_0000_rtsp.pcap",
    "capture_20250908_600sec_0000_http+mqtt+rtsp.pcap",
]
# Unique capture d'ATTAQUE : coupée dans le temps (début train / fin test)
FICHIER_ATTAQUE = "capture_20250909_600sec_0000_mirai-syn.pcap"
PART_ATTAQUE_TRAIN = 0.7

generateur = np.random.default_rng(GRAINE)


# =============================================================================
# Lecture d'un .pcap -> fenêtres
# =============================================================================
def pcap_vers_fenetres(chemin_pcap, max_fenetres=None):
    max_paquets = None if max_fenetres is None else max_fenetres * NB_PAQUETS
    lignes = []
    temps_precedent = None
    with PcapReader(chemin_pcap) as lecteur:
        for paquet in lecteur:
            taille = len(paquet)
            intervalle = 0.0 if temps_precedent is None else float(paquet.time) - temps_precedent
            temps_precedent = float(paquet.time)
            protocole, port = 0, 0
            if IP in paquet:
                protocole = int(paquet[IP].proto)
            if TCP in paquet:
                port = int(paquet[TCP].dport)
            elif UDP in paquet:
                port = int(paquet[UDP].dport)
            lignes.append([taille, intervalle, protocole, port])
            if max_paquets is not None and len(lignes) >= max_paquets:
                break
    lignes = np.array(lignes, dtype=np.float32)
    nb = len(lignes) // NB_PAQUETS
    return lignes[:nb * NB_PAQUETS].reshape(nb, NB_PAQUETS, NB_CARACTERISTIQUES)


def charger_captures(dossier, noms_fichiers, etiquette):
    """Charge une liste précise de fichiers, tous de la même classe."""
    morceaux = []
    for nom in noms_fichiers:
        fenetres = pcap_vers_fenetres(os.path.join(dossier, nom),
                                      max_fenetres=MAX_FENETRES_PAR_FICHIER)
        print(f"  {nom} -> {len(fenetres)} fenêtres")
        if len(fenetres) > 0:
            morceaux.append(fenetres)
    if morceaux:
        fenetres = np.concatenate(morceaux, axis=0)
    else:
        fenetres = np.empty((0, NB_PAQUETS, NB_CARACTERISTIQUES), dtype=np.float32)
    etiquettes = np.full(len(fenetres), float(etiquette), dtype=np.float32)
    return fenetres, etiquettes


# =============================================================================
# Normaliser / Modèle / Évaluer  (identiques à avant)
# =============================================================================
def normaliser(fenetres_train, fenetres_test):
    moyenne    = fenetres_train.reshape(-1, NB_CARACTERISTIQUES).mean(axis=0)
    ecart_type = fenetres_train.reshape(-1, NB_CARACTERISTIQUES).std(axis=0) + 1e-8
    return (fenetres_train - moyenne) / ecart_type, (fenetres_test - moyenne) / ecart_type


def construire_modele():
    modele = models.Sequential([
        layers.Input(shape=(NB_PAQUETS, NB_CARACTERISTIQUES)),
        layers.Conv1D(16, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(32, kernel_size=3, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    modele.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return modele


def evaluer(modele, fenetres_test, etiquettes_test):
    probabilites = modele.predict(fenetres_test, verbose=0).ravel()
    predictions       = (probabilites >= 0.5).astype(int)
    vraies_etiquettes = etiquettes_test.astype(int)

    vrais_positifs = int(np.sum((predictions == 1) & (vraies_etiquettes == 1)))
    vrais_negatifs = int(np.sum((predictions == 0) & (vraies_etiquettes == 0)))
    faux_positifs  = int(np.sum((predictions == 1) & (vraies_etiquettes == 0)))
    faux_negatifs  = int(np.sum((predictions == 0) & (vraies_etiquettes == 1)))

    precision  = vrais_positifs / (vrais_positifs + faux_positifs) if (vrais_positifs + faux_positifs) else 0.0
    rappel     = vrais_positifs / (vrais_positifs + faux_negatifs) if (vrais_positifs + faux_negatifs) else 0.0
    f1         = 2 * precision * rappel / (precision + rappel) if (precision + rappel) else 0.0
    exactitude = (vrais_positifs + vrais_negatifs) / len(vraies_etiquettes)

    print("\n" + "=" * 58)
    print("     RÉSULTATS SUR DES CAPTURES JAMAIS VUES (généralisation)")
    print("=" * 58)
    print("Matrice de confusion :")
    print("                  prédit NORMAL    prédit ATTAQUE")
    print(f"  réel NORMAL        VN={vrais_negatifs:<6}       FP={faux_positifs}")
    print(f"  réel ATTAQUE       FN={faux_negatifs:<6}       VP={vrais_positifs}")
    print("-" * 58)
    print(f"  Exactitude          : {exactitude:.3f}")
    print(f"  Précision           : {precision:.3f}")
    print(f"  Rappel (détection)  : {rappel:.3f}")
    print(f"  F1-score            : {f1:.3f}")
    print("=" * 58)

    # matrice de confusion graphique (vraies predictions sur les captures de test)
    tracer_matrice(vraies_etiquettes, predictions,
                   "Matrice de confusion — conception 1 (captures jamais vues)",
                   "matrice_confusion_conception1.png")
    plt.show()


def tracer_matrice(vrai, pred, titre, chemin):
    """Trace une matrice de confusion 2x2 (Benin / Attaque) et l'enregistre."""
    mc = confusion_matrix(vrai, pred, labels=[0, 1])
    classes = ["Bénin", "Attaque"]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(mc, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Vraie classe")
    ax.set_title(titre)
    total = mc.sum()
    seuil = mc.max() / 2.0
    for i in range(2):
        for j in range(2):
            n = mc[i, j]
            pourcent = f"\n{n/total:.1%}" if total else ""
            ax.text(j, i, f"{n}{pourcent}", ha="center", va="center",
                    color="white" if n > seuil else "#1F3A5F",
                    fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(chemin, dpi=150)
    print(f"  Matrice de confusion enregistrée : {chemin}")
    return mc


# =============================================================================
# main
# =============================================================================
def main():
    np.random.seed(GRAINE)
    tf.random.set_seed(GRAINE)

    print("--- Captures d'ENTRAÎNEMENT (normal) ---")
    fen_tr_normal, etq_tr_normal = charger_captures(DOSSIER_PCAP, CAPTURES_TRAIN_NORMAL, 0)
    print("--- Captures de TEST (normal, jamais vues) ---")
    fen_te_normal, etq_te_normal = charger_captures(DOSSIER_PCAP, CAPTURES_TEST_NORMAL, 0)

    print("--- Capture d'ATTAQUE (coupée dans le temps) ---")
    fen_attaque, etq_attaque = charger_captures(DOSSIER_PCAP, [FICHIER_ATTAQUE], 1)
    coupe = int(PART_ATTAQUE_TRAIN * len(fen_attaque))
    fen_tr_attaque, fen_te_attaque = fen_attaque[:coupe], fen_attaque[coupe:]   # début / fin
    etq_tr_attaque, etq_te_attaque = etq_attaque[:coupe], etq_attaque[coupe:]

    # assembler entraînement et test
    fenetres_train   = np.concatenate([fen_tr_normal, fen_tr_attaque], axis=0)
    etiquettes_train = np.concatenate([etq_tr_normal, etq_tr_attaque], axis=0)
    fenetres_test    = np.concatenate([fen_te_normal, fen_te_attaque], axis=0)
    etiquettes_test  = np.concatenate([etq_te_normal, etq_te_attaque], axis=0)

    # mélanger l'entraînement (sinon validation_split prendrait un bloc d'une seule classe)
    ordre = generateur.permutation(len(fenetres_train))
    fenetres_train, etiquettes_train = fenetres_train[ordre], etiquettes_train[ordre]

    print()
    print(f"Entraînement : {len(fenetres_train)} fenêtres "
          f"({int((etiquettes_train==0).sum())} normal / {int((etiquettes_train==1).sum())} attaque)")
    print(f"Test         : {len(fenetres_test)} fenêtres "
          f"({int((etiquettes_test==0).sum())} normal / {int((etiquettes_test==1).sum())} attaque)")
    print()

    fenetres_train, fenetres_test = normaliser(fenetres_train, fenetres_test)

    modele = construire_modele()
    modele.fit(fenetres_train, etiquettes_train,
               validation_split=0.1, epochs=15, batch_size=64, verbose=2)

    evaluer(modele, fenetres_test, etiquettes_test)


if __name__ == "__main__":
    main()
