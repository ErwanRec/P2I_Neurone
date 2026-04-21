import time
import json
import matplotlib.pyplot as plt
import document as dc
import fonction_RL as RL
from nn import MLP

start = time.time()

# ------------------------------------------------------------------ #
#  Hyperparamètres                                                     #
# ------------------------------------------------------------------ #
epsilon = 0.8
NB_EPOQUES = 1000

positions_initiales = {
    0: (45, 25), 1: (45, 30), # Equipe 1
    2: (55, 25), 3: (55, 30)  # Equipe 2
}

# Paramètres pour initialiser l'environnement Objet
PARAMS_MATCH = {
    'T': dc.T, 'K': dc.K, 'g': dc.g, 'mb': dc.mb, 'n': dc.n,
    'taille_terrain': dc.taille_terrain,
    'caracteristique': dc.caracteristique,
    'positions_initiales': positions_initiales
}


# ------------------------------------------------------------------ #
# Sauvegarde                                                           #
# ------------------------------------------------------------------ #

def sauvegarder_reseau(net, nom_fichier):
    """Sauvegarde les poids du réseau dans un fichier JSON."""
    # On extrait la valeur float de chaque objet Value
    poids = [p.data for p in net.parameters()]
    with open(nom_fichier, 'w') as f:
        json.dump(poids, f)
    print(f"Réseau sauvegardé dans {nom_fichier}")


# ------------------------------------------------------------------ #
#  Boucle principale                                                   #
# ------------------------------------------------------------------ #
net1 = MLP(48, [16, 16, 16, 1])
net2 = MLP(48, [16, 16, 16, 1])

historique = []

for epoque in range(NB_EPOQUES):
    traces1 = RL.init_traces(net1)
    traces2 = RL.init_traces(net2)

    # On récupère le 4eme argument retourné : la distance finale
    t1, t2, tb, dist = dc.match(net1, net2, traces1, traces2,PARAMS_MATCH, epsilon)

    historique.append(dist)

    end = time.time()
    print(f"Époque {epoque:3d} | Distance J0 = {dist:.2f} | {end - start:.1f}s")
    start = end

    if epoque % 5 == 0:
        print(f"  → Animation époque {epoque}")
        dc.afficher_match(t1, t2, tb)

    epsilon = max(0.01, epsilon * 0.995)
sauvegarder_reseau(net1, "ia_equipe1_entrainee.json")
sauvegarder_reseau(net2, "ia_equipe2_entrainee.json")
plt.figure()
plt.plot(historique)
plt.title("Progression de l'IA (Version Objet) — Distance finale")
plt.xlabel("Époque")
plt.ylabel("Position X")
plt.grid(True)
plt.tight_layout()
plt.show()
