"""
=============================================================================
 DIAGNOSTIC — les faux positifs sont-ils des fenêtres À CHEVAL ?
=============================================================================
 Hypothèse : une partie des "faux positifs" ne sont pas de vraies erreurs,
 mais des fenêtres de TRANSITION (bénignes à la majorité, mais contenant
 déjà quelques paquets d'attaque). Elles ressemblent légitimement à des
 attaques -> le modèle a "presque raison".

 On classe chaque faux positif selon le nombre de paquets d'attaque qu'il
 contient (entre 0 et 12, puisqu'au-delà de 12 la fenêtre serait étiquetée
 attaque) :
   - PURE bénigne      : 0 paquet d'attaque
   - À CHEVAL          : 1 à 12 paquets d'attaque

 On affiche aussi la proportion de SYN de chaque sous-groupe, pour voir si
 les SYN des faux positifs viennent des paquets d'attaque (fenêtres à cheval)
 ou d'un vrai trafic SYN bénin.
=============================================================================
"""

import os
import glob
import ipaddress
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scapy.all import IP, TCP, UDP, PcapReader

from cnn_ddos_crossval import (
    normaliser, construire_modele, DOSSIERS_PCAP,
    NB_PAQUETS, NB_CARACTERISTIQUES, MAX_FENETRES_PAR_FICHIER,
    SOUS_RESEAU_ATTAQUE, GRAINE,
)


def lire(chemin, max_fenetres):
    max_paquets = max_fenetres * NB_PAQUETS
    feats, etats, est_syn = [], [], []
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
            syn = 1 if (TCP in p and (int(p[TCP].flags) & 0x02) and not (int(p[TCP].flags) & 0x10)) else 0
            est_syn.append(syn)
            if len(feats) >= max_paquets:
                break
    n = len(feats) // NB_PAQUETS
    feats = np.array(feats, np.float32)[:n*NB_PAQUETS].reshape(n, NB_PAQUETS, NB_CARACTERISTIQUES)
    etats = np.array(etats, np.int32)[:n*NB_PAQUETS].reshape(n, NB_PAQUETS)
    est_syn = np.array(est_syn, np.int32)[:n*NB_PAQUETS].reshape(n, NB_PAQUETS)
    etiquettes = (etats.sum(axis=1) > (NB_PAQUETS // 2)).astype(np.float32)
    nb_att_par_fenetre = etats.sum(axis=1)        # combien de paquets d'attaque dans chaque fenêtre
    prop_syn = est_syn.mean(axis=1)
    return feats, etiquettes, nb_att_par_fenetre, prop_syn


def main():
    tf.keras.utils.set_random_seed(GRAINE)
    print("=== Chargement ===")
    F, Y, A, S = [], [], [], []
    for dossier in DOSSIERS_PCAP:
        for chemin in sorted(glob.glob(os.path.join(dossier, "*.pcap"))):
            f, y, a, s = lire(chemin, MAX_FENETRES_PAR_FICHIER)
            print(f"  {os.path.basename(chemin)} -> {len(f)} fenêtres")
            F.append(f); Y.append(y); A.append(a); S.append(s)
    fenetres = np.concatenate(F); etiquettes = np.concatenate(Y)
    nb_att = np.concatenate(A);   prop_syn = np.concatenate(S)

    idx = np.arange(len(fenetres))
    itr, ite = train_test_split(idx, test_size=0.25, stratify=etiquettes, random_state=GRAINE)
    ftr, fte = normaliser(fenetres[itr], fenetres[ite])
    modele = construire_modele()
    modele.fit(ftr, etiquettes[itr], epochs=15, batch_size=64, verbose=0)
    pred = (modele.predict(fte, verbose=0).ravel() >= 0.5).astype(int)
    vrai = etiquettes[ite].astype(int)

    nb_att_te = nb_att[ite]
    syn_te = prop_syn[ite]

    fp = (pred == 1) & (vrai == 0)              # faux positifs
    fp_pures   = fp & (nb_att_te == 0)          # 0 paquet d'attaque -> vraie erreur
    fp_cheval  = fp & (nb_att_te > 0)           # contient de l'attaque -> fenêtre de transition

    total = int(fp.sum())
    print(f"\nFaux positifs : {total}")
    if total:
        print(f"  dont À CHEVAL (>=1 paquet d'attaque) : {int(fp_cheval.sum())}  "
              f"({100*fp_cheval.sum()/total:.0f}%)")
        print(f"  dont PURES bénignes (0 attaque)      : {int(fp_pures.sum())}  "
              f"({100*fp_pures.sum()/total:.0f}%)")
        print("\nProportion de SYN par sous-groupe :")
        if fp_cheval.any():
            print(f"  faux positifs à cheval : {syn_te[fp_cheval].mean():.3f}  "
                  f"(nb paquets d'attaque moyen : {nb_att_te[fp_cheval].mean():.1f}/24)")
        if fp_pures.any():
            print(f"  faux positifs purs     : {syn_te[fp_pures].mean():.3f}")
        print("\nLecture :")
        print("  - beaucoup à cheval -> le souci est la RÈGLE d'étiquetage de fenêtre,")
        print("    pas les caractéristiques.")
        print("  - beaucoup de purs avec SYN élevé -> vrai trafic SYN bénin : le flag")
        print("    SYN ne suffira pas seul.")


if __name__ == "__main__":
    main()
