"""
=============================================================================
 CNN 1D - SEUIL D'ÉTIQUETAGE DE FENÊTRE réglable + comparaison
=============================================================================
 Le problème identifié : la règle "fenêtre = attaque si > 12 paquets d'attaque
 sur 24" étiquette "bénignes" des fenêtres de transition qui contiennent
 déjà beaucoup de flood (~10 paquets). Le modèle les détecte comme attaques,
 ce qui gonfle artificiellement les "faux positifs".

 Ici, le SEUIL devient un paramètre. On compare plusieurs seuils :
   - seuil bas  -> on étiquette attaque dès quelques paquets de flood
                   => détection précoce, moins de fenêtres à cheval mal comptées
   - seuil haut -> étiquette plus "pure" mais détection tardive

 Pour chaque seuil : validation croisée 5 plis, moyenne des métriques,
 et proportion des faux positifs qui sont en réalité des fenêtres À CHEVAL.

 On NE retire PAS les fenêtres à cheval (ce serait cacher un scénario réel) :
 on garde toutes les fenêtres, on ajuste seulement le seuil qui définit
 l'étiquette.
=============================================================================
"""

import os
import glob
import ipaddress
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from scapy.all import IP, TCP, UDP, PcapReader

GRAINE = 42
NB_PAQUETS = 24
NB_CARACTERISTIQUES = 4
MAX_FENETRES_PAR_FICHIER = 2000
NB_PLIS = 5
SEUILS_A_TESTER = [1, 4, 6, 8, 12]        # nb minimal de paquets d'attaque pour étiqueter "attaque"

DOSSIERS_PCAP = [
    r"C:\Users\jojo_\Desktop\data_pcap_bachelor_2",
]
SOUS_RESEAU_ATTAQUE = ipaddress.ip_network("192.168.10.100/30")


def lire_pcap(chemin, max_fenetres):
    """Renvoie (fenetres, nb_attaque_par_fenetre).
    On garde le NOMBRE de paquets d'attaque par fenêtre : l'étiquette sera
    dérivée ensuite selon le seuil, sans relire le fichier."""
    max_paquets = max_fenetres * NB_PAQUETS
    feats, etats = [], []
    t_prec = None
    with PcapReader(chemin) as lecteur:
        for p in lecteur:
            taille = len(p)
            interv = 0.0 if t_prec is None else float(p.time) - t_prec
            t_prec = float(p.time)
            proto = int(p[IP].proto) if IP in p else 0
            port = int(p[TCP].dport) if TCP in p else (int(p[UDP].dport) if UDP in p else 0)
            feats.append([taille, interv, proto, port])
            attaque = IP in p and ipaddress.ip_address(p[IP].dst) in SOUS_RESEAU_ATTAQUE
            etats.append(1 if attaque else 0)
            if len(feats) >= max_paquets:
                break
    n = len(feats) // NB_PAQUETS
    feats = np.array(feats, np.float32)[:n*NB_PAQUETS].reshape(n, NB_PAQUETS, NB_CARACTERISTIQUES)
    etats = np.array(etats, np.int32)[:n*NB_PAQUETS].reshape(n, NB_PAQUETS)
    return feats, etats.sum(axis=1)          # (fenetres, nb paquets d'attaque par fenêtre)


def charger():
    F, A = [], []
    for dossier in DOSSIERS_PCAP:
        for chemin in sorted(glob.glob(os.path.join(dossier, "*.pcap"))):
            f, a = lire_pcap(chemin, MAX_FENETRES_PAR_FICHIER)
            print(f"  {os.path.basename(chemin)} -> {len(f)} fenêtres")
            F.append(f); A.append(a)
    return np.concatenate(F), np.concatenate(A)


def normaliser(tr, te):
    moy = tr.reshape(-1, NB_CARACTERISTIQUES).mean(axis=0)
    ect = tr.reshape(-1, NB_CARACTERISTIQUES).std(axis=0) + 1e-8
    return (tr - moy) / ect, (te - moy) / ect


def construire_modele():
    m = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(NB_PAQUETS, NB_CARACTERISTIQUES)),
        tf.keras.layers.Conv1D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


def evaluer_seuil(fenetres, nb_attaque, seuil):
    """Validation croisée pour un seuil donné. Renvoie les moyennes des
    métriques et la proportion de faux positifs qui sont des fenêtres à cheval."""
    etiquettes = (nb_attaque >= seuil).astype(np.float32)

    decoupage = StratifiedKFold(n_splits=NB_PLIS, shuffle=True, random_state=GRAINE)
    scores, fp_total, fp_cheval_total = [], 0, 0

    for idx_tr, idx_te in decoupage.split(fenetres, etiquettes):
        ftr, fte = normaliser(fenetres[idx_tr], fenetres[idx_te])
        modele = construire_modele()
        modele.fit(ftr, etiquettes[idx_tr], epochs=15, batch_size=64, verbose=0)
        pred = (modele.predict(fte, verbose=0).ravel() >= 0.5).astype(int)
        vrai = etiquettes[idx_te].astype(int)
        att_te = nb_attaque[idx_te]

        vp = int(np.sum((pred==1)&(vrai==1))); vn = int(np.sum((pred==0)&(vrai==0)))
        fp = int(np.sum((pred==1)&(vrai==0))); fn = int(np.sum((pred==0)&(vrai==1)))
        precision = vp/(vp+fp) if (vp+fp) else 0.0
        rappel    = vp/(vp+fn) if (vp+fn) else 0.0
        f1 = 2*precision*rappel/(precision+rappel) if (precision+rappel) else 0.0
        scores.append((precision, rappel, f1))

        masque_fp = (pred==1)&(vrai==0)
        fp_total += int(masque_fp.sum())
        fp_cheval_total += int(((att_te>0)&masque_fp).sum())   # faux positifs contenant du flood

    scores = np.array(scores).mean(axis=0)
    part_cheval = (100*fp_cheval_total/fp_total) if fp_total else 0.0
    return scores, fp_total//NB_PLIS, part_cheval


def main():
    tf.keras.utils.set_random_seed(GRAINE)
    print("=== Chargement ===")
    fenetres, nb_attaque = charger()
    print(f"\nTotal : {len(fenetres)} fenêtres\n")

    print("=" * 78)
    print(f"{'seuil':>6}{'attaque/benin':>16}{'precision':>11}{'rappel':>9}{'F1':>7}"
          f"{'FP/pli':>8}{'FP a cheval':>13}")
    print("=" * 78)
    for seuil in SEUILS_A_TESTER:
        etq = (nb_attaque >= seuil).astype(int)
        (pre, rap, f1), fp_pli, part = evaluer_seuil(fenetres, nb_attaque, seuil)
        print(f"{seuil:>6}{f'{int(etq.sum())}/{int((etq==0).sum())}':>16}"
              f"{pre:>11.3f}{rap:>9.3f}{f1:>7.3f}{fp_pli:>8}{part:>11.0f}%")
    print("=" * 78)
    print("\nLecture : un seuil plus bas fait basculer les fenêtres à cheval du côté")
    print("'attaque' -> la précision remonte (moins de faux positifs), au prix")
    print("d'étiqueter comme attaque des fenêtres à peine touchées par le flood.")


if __name__ == "__main__":
    main()
