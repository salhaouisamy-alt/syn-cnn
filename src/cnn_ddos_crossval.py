"""
=============================================================================
 CNN 1D - Détection SYN flood  ·  étiquetage par paquet + validation croisée
=============================================================================

 Nouveautés par rapport à la version précédente :
   1. On lit DEUX dossiers de captures (anciennes + nouvelles).
   2. L'étiquette est déduite PAR PAQUET de l'IP de destination :
      un paquet vers le sous-réseau 192.168.10.100/30 est une ATTAQUE.
   3. Une fenêtre hérite de son étiquette à la MAJORITÉ de ses paquets.
   4. Évaluation par VALIDATION CROISÉE à 5 plis (moyenne ± écart-type),
      sur des fenêtres mélangées, avec normalisation recalculée à chaque pli.

 Caractéristiques inchangées : taille, intervalle, protocole, port.
 L'IP de destination sert UNIQUEMENT à étiqueter ; elle n'est jamais donnée
 au modèle (sinon il recopierait la règle d'étiquetage).

 Périmètre : SYN flood, aucun filtrage TCP pour l'instant (tout est gardé).

 Lancement :
     python cnn_ddos_crossval.py
=============================================================================
"""

import os
import glob
import ipaddress
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from scapy.all import IP, TCP, UDP, PcapReader
import matplotlib.pyplot as plt

# --- Configuration ---
GRAINE = 42
NB_PAQUETS = 24
NB_CARACTERISTIQUES = 4
TAILLE, INTERVALLE, PROTOCOLE, PORT = 0, 1, 2, 3
MAX_FENETRES_PAR_FICHIER = 2000
NB_PLIS = 5

# Seuil d'étiquetage de la fenêtre : nombre minimal de paquets d'attaque (sur
# NB_PAQUETS) pour qu'une fenêtre soit étiquetée "attaque". On l'a fixé à 6
# après comparaison (voir cnn_ddos_seuil.py) : un seuil de 12 (majorité stricte)
# étiquetait "bénignes" des fenêtres de transition contenant déjà ~10 paquets de
# flood, que le modèle détectait à juste titre -> faux positifs artificiels.
SEUIL_ETIQUETTE = 6

# Les DEUX dossiers de captures à traiter ensemble
DOSSIERS_PCAP = [
    r"C:\Users\jojo_\Desktop\data_pcap_bachelor_2",
]

# Sous-réseau de destination = trafic d'attaque (Mirai SYN flood)
SOUS_RESEAU_ATTAQUE = ipaddress.ip_network("192.168.10.100/30")

generateur = np.random.default_rng(GRAINE)


# =============================================================================
# Étiquetage d'un paquet à partir de son IP de destination
# =============================================================================
def paquet_est_attaque(paquet):
    """Renvoie True si l'IP de destination du paquet appartient au sous-réseau
    d'attaque. Un paquet sans couche IP est considéré bénin."""
    if IP not in paquet:
        return False
    try:
        return ipaddress.ip_address(paquet[IP].dst) in SOUS_RESEAU_ATTAQUE
    except ValueError:
        return False


# =============================================================================
# Lecture d'un .pcap -> fenêtres + étiquette de fenêtre (règle de majorité)
# =============================================================================
def pcap_vers_fenetres(chemin_pcap, max_fenetres=None):
    """Lit un .pcap et renvoie :
         fenetres   : tableau (nb_fenetres, NB_PAQUETS, NB_CARACTERISTIQUES)
         etiquettes : tableau (nb_fenetres,) -> 1 attaque / 0 bénin
    L'étiquette de chaque paquet vient de son IP destination ; l'étiquette de
    la fenêtre est fixée à la MAJORITÉ des étiquettes de ses 24 paquets.
    """
    max_paquets = None if max_fenetres is None else max_fenetres * NB_PAQUETS
    lignes = []            # caractéristiques : [taille, intervalle, protocole, port] par paquet
    etats  = []            # étiquette de chaque paquet : 1 attaque / 0 bénin
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
            etats.append(1 if paquet_est_attaque(paquet) else 0)

            if max_paquets is not None and len(lignes) >= max_paquets:
                break

    lignes = np.array(lignes, dtype=np.float32)
    etats  = np.array(etats, dtype=np.int32)

    nb = len(lignes) // NB_PAQUETS
    lignes = lignes[:nb * NB_PAQUETS].reshape(nb, NB_PAQUETS, NB_CARACTERISTIQUES)
    etats  = etats[:nb * NB_PAQUETS].reshape(nb, NB_PAQUETS)

    # étiquette de fenêtre : "attaque" si au moins SEUIL_ETIQUETTE de ses paquets
    # sont des attaques (voir la justification près de la constante)
    etiquettes = (etats.sum(axis=1) >= SEUIL_ETIQUETTE).astype(np.float32)
    return lignes, etiquettes


def charger_dossiers(dossiers):
    """Lit tous les .pcap de plusieurs dossiers et empile leurs fenêtres.
    Renvoie (toutes_fenetres, toutes_etiquettes, nb_fenetres_mixtes)."""
    liste_fenetres, liste_etiquettes = [], []

    for dossier in dossiers:
        for chemin in sorted(glob.glob(os.path.join(dossier, "*.pcap"))):
            fenetres, etiquettes = pcap_vers_fenetres(chemin, MAX_FENETRES_PAR_FICHIER)
            nb_att = int(etiquettes.sum())
            print(f"  {os.path.basename(chemin)} -> {len(fenetres)} fenêtres "
                  f"({nb_att} attaque / {len(fenetres) - nb_att} bénin)")
            if len(fenetres) > 0:
                liste_fenetres.append(fenetres)
                liste_etiquettes.append(etiquettes)

    fenetres   = np.concatenate(liste_fenetres, axis=0)
    etiquettes = np.concatenate(liste_etiquettes, axis=0)
    return fenetres, etiquettes


# =============================================================================
# Normalisation / Modèle / Métriques
# =============================================================================
def normaliser(fenetres_train, fenetres_test):
    """Standardise avec moyenne/écart-type calculés SUR LE TRAIN uniquement.
    (Recalculé à chaque pli pour éviter toute fuite du pli de test.)"""
    moyenne    = fenetres_train.reshape(-1, NB_CARACTERISTIQUES).mean(axis=0)
    ecart_type = fenetres_train.reshape(-1, NB_CARACTERISTIQUES).std(axis=0) + 1e-8
    return (fenetres_train - moyenne) / ecart_type, (fenetres_test - moyenne) / ecart_type


def construire_modele():
    """Le CNN 1D léger (identique aux versions précédentes)."""
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


def metriques(modele, fenetres_test, etiquettes_test):
    """Renvoie (exactitude, precision, rappel, f1, FP, FN) sur un pli."""
    proba = modele.predict(fenetres_test, verbose=0).ravel()
    pred  = (proba >= 0.5).astype(int)
    vrai  = etiquettes_test.astype(int)

    vp = int(np.sum((pred == 1) & (vrai == 1)))
    vn = int(np.sum((pred == 0) & (vrai == 0)))
    fp = int(np.sum((pred == 1) & (vrai == 0)))
    fn = int(np.sum((pred == 0) & (vrai == 1)))

    precision = vp / (vp + fp) if (vp + fp) else 0.0
    rappel    = vp / (vp + fn) if (vp + fn) else 0.0
    f1 = 2 * precision * rappel / (precision + rappel) if (precision + rappel) else 0.0
    exactitude = (vp + vn) / len(vrai)
    return exactitude, precision, rappel, f1, fp, fn, pred, vrai


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
            ax.text(j, i, f"{n}\n{n/total:.1%}", ha="center", va="center",
                    color="white" if n > seuil else "#1F3A5F",
                    fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(chemin, dpi=150)
    print(f"  Matrice enregistrée : {chemin}")
    return mc


def tracer_matrices_par_pli(matrices, seuil_etq, chemin, cols=2):
    """Trace les matrices de confusion des plis sur une grille (cols par ligne)."""
    import math
    n = len(matrices)
    cols = min(cols, n)
    lignes = math.ceil(n / cols)
    fig, axes = plt.subplots(lignes, cols, figsize=(3.6 * cols, 3.4 * lignes))
    axes = np.array(axes).reshape(-1)   # aplatir pour indexer simplement
    classes = ["Bénin", "Attaque"]
    for k, (vrai, pred) in enumerate(matrices):
        mc = confusion_matrix(vrai, pred, labels=[0, 1])
        ax = axes[k]
        im = ax.imshow(mc, cmap="Blues")
        ax.set_title(f"Pli {k + 1}")
        ax.set_xticks([0, 1]); ax.set_xticklabels(classes, fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Prédite", fontsize=9)
        if k % cols == 0:
            ax.set_ylabel("Vraie classe", fontsize=9)
        s = mc.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, mc[i, j], ha="center", va="center",
                        color="white" if mc[i, j] > s else "#1F3A5F",
                        fontsize=10, fontweight="bold")
    for k in range(n, len(axes)):   # masquer les cases vides eventuelles
        axes[k].axis("off")
    fig.suptitle(f"Matrices de confusion par pli (seuil d'étiquetage = {seuil_etq})")
    plt.tight_layout()
    plt.savefig(chemin, dpi=150)
    print(f"  Matrices par pli enregistrées : {chemin}")


# =============================================================================
# main - validation croisée à 5 plis
# =============================================================================
def main():
    tf.keras.utils.set_random_seed(GRAINE)

    print("=== Chargement des captures ===")
    fenetres, etiquettes = charger_dossiers(DOSSIERS_PCAP)
    nb_att = int(etiquettes.sum())
    print(f"\nTotal : {len(fenetres)} fenêtres  "
          f"({nb_att} attaque / {len(fenetres) - nb_att} bénin)\n")

    # StratifiedKFold garde la même proportion attaque/bénin dans chaque pli
    decoupage = StratifiedKFold(n_splits=NB_PLIS, shuffle=True, random_state=GRAINE)

    scores = []           # (exactitude, precision, rappel, f1) pour chaque pli
    vrais_tous = []       # vraies etiquettes hors-pli (toutes les fenetres)
    preds_tous = []       # predictions hors-pli (toutes les fenetres)
    matrices_plis = []    # (vrai, pred) de chaque pli, pour les matrices par pli
    for num_pli, (idx_train, idx_test) in enumerate(decoupage.split(fenetres, etiquettes), 1):
        fen_tr, fen_te = fenetres[idx_train], fenetres[idx_test]
        etq_tr, etq_te = etiquettes[idx_train], etiquettes[idx_test]

        fen_tr, fen_te = normaliser(fen_tr, fen_te)   # recalcul par pli

        modele = construire_modele()
        modele.fit(fen_tr, etq_tr, epochs=15, batch_size=64, verbose=0)

        exa, pre, rap, f1, fp, fn, pred_pli, vrai_pli = metriques(modele, fen_te, etq_te)
        scores.append((exa, pre, rap, f1))
        vrais_tous.append(vrai_pli)
        preds_tous.append(pred_pli)
        matrices_plis.append((vrai_pli, pred_pli))
        print(f"Pli {num_pli}/{NB_PLIS} : exactitude={exa:.3f}  precision={pre:.3f}  "
              f"rappel={rap:.3f}  F1={f1:.3f}  (FP={fp}, FN={fn})")

    scores = np.array(scores)
    moyennes = scores.mean(axis=0)
    ecarts   = scores.std(axis=0)

    print("\n" + "=" * 58)
    print(f"   VALIDATION CROISÉE {NB_PLIS} PLIS  (moyenne ± écart-type)")
    print("=" * 58)
    for i, nom in enumerate(["Exactitude", "Précision ", "Rappel    ", "F1-score  "]):
        print(f"  {nom} : {moyennes[i]:.3f} ± {ecarts[i]:.3f}")
    print("=" * 58)

    # -------------------------------------------------------------------------
    # Matrices de confusion (vraies, construites a partir des predictions)
    # Chaque fenetre est testee exactement une fois (validation croisee),
    # on agrege donc toutes les predictions hors-pli en une seule matrice.
    # -------------------------------------------------------------------------
    print("\n=== Matrices de confusion ===")
    vrais_tous = np.concatenate(vrais_tous)
    preds_tous = np.concatenate(preds_tous)
    tracer_matrice(
        vrais_tous, preds_tous,
        f"Matrice de confusion — validation croisée (seuil {SEUIL_ETIQUETTE})",
        "matrice_confusion_globale.png",
    )
    tracer_matrices_par_pli(matrices_plis, SEUIL_ETIQUETTE, "matrice_confusion_par_pli.png")
    plt.show()


if __name__ == "__main__":
    main()