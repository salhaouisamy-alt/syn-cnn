"""
=============================================================================
 DIAGNOSTIC ÉLARGI — quelle information nouvelle sépare le bénin de l'attaque ?
=============================================================================
 On reprend le modèle actuel (4 caractéristiques), on repère les faux positifs
 (bénin classé attaque), et pour trois groupes — faux positifs, bénin bien
 classé, vraies attaques — on mesure DEUX pistes candidates :

   1. proportion de paquets SYN dans la fenêtre  (drapeau TCP "S")
   2. nombre d'adresses IP sources distinctes dans la fenêtre

 Objectif : voir laquelle de ces deux informations sépare nettement les faux
 positifs des vraies attaques. Celle qui sépare est la caractéristique à
 ajouter au modèle.

 Nécessite que les captures gardent l'IP source et les flags -> on RELIT les
 pcaps ici en extrayant ces deux infos par paquet (le modèle, lui, ne les
 reçoit pas : c'est juste un diagnostic).
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


def lire_pcap_complet(chemin, max_fenetres):
    """Relit un pcap en extrayant, par fenêtre de NB_PAQUETS :
       - les 4 caractéristiques habituelles (pour rejouer le modèle)
       - l'étiquette (majorité, via IP destination)
       - la proportion de paquets SYN
       - le nombre d'IP sources distinctes
    """
    max_paquets = max_fenetres * NB_PAQUETS
    feats, etats, est_syn, src = [], [], [], []
    t_prec = None

    with PcapReader(chemin) as lecteur:
        for p in lecteur:
            taille = len(p)
            interv = 0.0 if t_prec is None else float(p.time) - t_prec
            t_prec = float(p.time)
            proto = int(p[IP].proto) if IP in p else 0
            port = int(p[TCP].dport) if TCP in p else (int(p[UDP].dport) if UDP in p else 0)
            feats.append([taille, interv, proto, port])

            # étiquette du paquet (IP destination dans le sous-réseau d'attaque)
            attaque = IP in p and ipaddress.ip_address(p[IP].dst) in SOUS_RESEAU_ATTAQUE
            etats.append(1 if attaque else 0)

            # piste 1 : le paquet porte-t-il le drapeau SYN (et pas ACK) ?
            syn = 1 if (TCP in p and (int(p[TCP].flags) & 0x02) and not (int(p[TCP].flags) & 0x10)) else 0
            est_syn.append(syn)

            # piste 2 : IP source (pour compter les distinctes ensuite)
            src.append(p[IP].src if IP in p else "?")

            if len(feats) >= max_paquets:
                break

    n = len(feats) // NB_PAQUETS
    feats = np.array(feats, np.float32)[:n * NB_PAQUETS].reshape(n, NB_PAQUETS, NB_CARACTERISTIQUES)
    etats = np.array(etats, np.int32)[:n * NB_PAQUETS].reshape(n, NB_PAQUETS)
    est_syn = np.array(est_syn, np.int32)[:n * NB_PAQUETS].reshape(n, NB_PAQUETS)
    src = np.array(src, dtype=object)[:n * NB_PAQUETS].reshape(n, NB_PAQUETS)

    etiquettes = (etats.sum(axis=1) > (NB_PAQUETS // 2)).astype(np.float32)
    prop_syn = est_syn.mean(axis=1)                                   # proportion SYN par fenêtre
    nb_sources = np.array([len(set(ligne)) for ligne in src])         # sources distinctes par fenêtre
    return feats, etiquettes, prop_syn, nb_sources


def main():
    tf.keras.utils.set_random_seed(GRAINE)

    print("=== Chargement (avec SYN et sources) ===")
    F, Y, S, N = [], [], [], []
    for dossier in DOSSIERS_PCAP:
        for chemin in sorted(glob.glob(os.path.join(dossier, "*.pcap"))):
            f, y, s, n = lire_pcap_complet(chemin, MAX_FENETRES_PAR_FICHIER)
            print(f"  {os.path.basename(chemin)} -> {len(f)} fenêtres")
            F.append(f); Y.append(y); S.append(s); N.append(n)

    fenetres   = np.concatenate(F)
    etiquettes = np.concatenate(Y)
    prop_syn   = np.concatenate(S)
    nb_sources = np.concatenate(N)

    # entraîner le modèle actuel (4 features) et repérer ses faux positifs
    idx = np.arange(len(fenetres))
    itr, ite = train_test_split(idx, test_size=0.25, stratify=etiquettes, random_state=GRAINE)
    ftr, fte = normaliser(fenetres[itr], fenetres[ite])
    modele = construire_modele()
    modele.fit(ftr, etiquettes[itr], epochs=15, batch_size=64, verbose=0)
    pred = (modele.predict(fte, verbose=0).ravel() >= 0.5).astype(int)
    vrai = etiquettes[ite].astype(int)

    # sur le jeu de test : SYN et sources correspondants
    syn_te = prop_syn[ite]
    src_te = nb_sources[ite]

    fp = (pred == 1) & (vrai == 0)   # faux positifs
    ok = (pred == 0) & (vrai == 0)   # bénin bien classé
    at = (vrai == 1)                 # vraies attaques

    def moy(valeurs, masque, fmt):
        return format(valeurs[masque].mean(), fmt) if masque.any() else "-"

    print(f"\nTest : {int(fp.sum())} faux positifs, {int(ok.sum())} bénin OK, {int(at.sum())} attaques\n")
    print("=" * 74)
    print(f"{'Piste candidate':38}{'FAUX POSITIFS':>13}{'bénin OK':>11}{'attaques':>11}")
    print("=" * 74)
    print(f"{'proportion de paquets SYN':38}"
          f"{moy(syn_te, fp, '.3f'):>13}{moy(syn_te, ok, '.3f'):>11}{moy(syn_te, at, '.3f'):>11}")
    print(f"{'nombre d IP sources distinctes':38}"
          f"{moy(src_te, fp, '.2f'):>13}{moy(src_te, ok, '.2f'):>11}{moy(src_te, at, '.2f'):>11}")
    print("=" * 74)
    print("\nLecture : la BONNE piste est celle où FAUX POSITIFS ressemble à")
    print("bénin OK et DIFFÈRE des attaques. C'est elle qui séparerait les")
    print("fenêtres que le modèle confond aujourd'hui.")


if __name__ == "__main__":
    main()
