import time
import numpy as np
import matplotlib.pyplot as plt

import document as dc
import fonction_RL as RL
from nn import MLP

start = time.time()

# ------------------------------------------------------------------ #
#  Hyperparamètres                                                     #
# ------------------------------------------------------------------ #
epsilon = 0.8
NB_EPOQUES = 120

# Paramètres pour initialiser l'environnement Objet
PARAMS_MATCH = {
    'T': dc.T, 'K': dc.K, 'g': dc.g, 'mb': dc.mb, 'n': dc.n,
    'taille_terrain': dc.taille_terrain,
    'caracteristique': dc.caracteristique
}

# ------------------------------------------------------------------ #
#  Sélection d'action                                                  #
# ------------------------------------------------------------------ #
def choisir_meilleure_action(env, S: np.ndarray, net: MLP, equipe: int):
    actions_possibles = RL.actions(env, equipe)

    # Élagage directionnel
    actions_filtrees = []
    for act in actions_possibles:
        fx = act[0] if not isinstance(act[0], list) else act[0][0]
        if equipe == 1 and fx < 0:
            continue
        if equipe == 2 and fx > 0:
            continue
        actions_filtrees.append(act)

    if not actions_filtrees:
        actions_filtrees = actions_possibles

    nb   = int(len(env.joueurs) / 2)
    zero = [0, 0, [0, 0, 0]]

    etats_suivants = []
    for act in actions_filtrees:
        a1 = [act] + [zero] * (nb - 1)
        a2 = [zero] * nb
        etats_suivants.append(RL.etat(env, a1, a2))

    scores   = RL.choix_batch(net, etats_suivants)
    meilleur = int(np.argmax(scores))
    return actions_filtrees[meilleur]

# ------------------------------------------------------------------ #
#  Un match complet (Boucle d'environnement)                           #
# ------------------------------------------------------------------ #
def match(net1: MLP, net2: MLP, traces1: dict, traces2: dict):
    env = dc.Match(PARAMS_MATCH)
    env.balle.porteur = 0 # Le joueur 0 commence avec la balle
    
    nb = int(len(env.joueurs) / 2)
    zero = [0, 0, [0, 0, 0]]
    
    # Trajectoires joueurs
    traj1 = {"x": [[] for _ in range(nb)], "y": [[] for _ in range(nb)]}
    traj2 = {"x": [[] for _ in range(nb)], "y": [[] for _ in range(nb)]}
 
    # Trajectoire balle + porteur à chaque step
    traj_balle = {"x": [], "y": [], "porteur": []}

    while not env.etat_final():
        t_step = env.t
        S = RL.recuperer_etat_au_temps_t(env, t_step)

        # --- Choix actions ---
        if np.random.rand() < epsilon:
            action1 = RL.actions(env, 1)[np.random.randint(len(RL.actions(env, 1)))]
        else:
            action1 = choisir_meilleure_action(env, S, net1, equipe=1)

        if np.random.rand() < epsilon:
            action2 = RL.actions(env, 2)[np.random.randint(len(RL.actions(env, 2)))]
        else:
            action2 = choisir_meilleure_action(env, S, net2, equipe=2)

        a1_liste = [action1] + [zero] * (nb - 1)
        a2_liste = [action2] + [zero] * (nb - 1)

        # --- Étape de simulation (OOP) ---
        env.step(a1_liste, a2_liste)

        # --- État suivant et récompense ---
        S_next = RL.recuperer_etat_au_temps_t(env, env.t)
        final  = env.etat_final()
        R      = RL.calculer_recompense(env, final)

        # --- Mise à jour TD(λ) ---
        RL.TD_lambda(net1, traces1, S, S_next,  R)
        RL.TD_lambda(net2, traces2, S, S_next, -R)

    # --- Extraction pour l'affichage ---
    end_t = env.t + 1
    traj1 = {"x": [], "y": []}
    traj2 = {"x": [], "y": []}
    
    for i in range(nb):
        traj1["x"].append(env.joueurs[i].pos_x[:end_t])
        traj1["y"].append(env.joueurs[i].pos_y[:end_t])
        traj2["x"].append(env.joueurs[nb+i].pos_x[:end_t])
        traj2["y"].append(env.joueurs[nb+i].pos_y[:end_t])

    pid = env.balle.porteur
    if pid != -1:
        bx = env.joueurs[pid].pos_x[env.t]
        by = env.joueurs[pid].pos_y[env.t]
    else:
        bx = env.balle.x[env.t]
        by = env.balle.y[env.t]

    traj_balle["x"].append(bx)
    traj_balle["y"].append(by)
    traj_balle["porteur"].append(pid)
    
    dist_finale = traj1["x"][0][-1] # Distance du J0
    return traj1, traj2, traj_balle, dist_finale

# ------------------------------------------------------------------ #
#  Affichage                                                           #
# ------------------------------------------------------------------ #
def afficher_match(traj1, traj2, traj_balle):
    print(traj_balle)
    plt.figure(figsize=(12, 7))
    nb_steps = len(traj1["x"][0])
    from matplotlib.lines import Line2D
    legende = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue',
               markersize=12, markeredgecolor='black', label='Équipe 1'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato',
               markersize=12, markeredgecolor='black', label='Équipe 2'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
               markersize=10, markeredgecolor='black', label='Balle'),
    ]
    
    for i in range(nb_steps):
        plt.clf()
        plt.xlim(0, dc.taille_terrain["longeur"])
        plt.ylim(0, dc.taille_terrain["largeur"])
        plt.axvline(x=dc.taille_terrain["longeur"], color='green', lw=5, alpha=0.5)
        # ── Équipe 1 (bleu) ──────────────────────────────────────────
        for j in range(len(traj1["x"])):
            if i < len(traj1["x"][j]):
                plt.plot(traj1["x"][j][:i+1], traj1["y"][j][:i+1],
                         color='royalblue', alpha=0.25, lw=1.5)
                plt.scatter(traj1["x"][j][i], traj1["y"][j][i],
                            color='royalblue', s=200,
                            edgecolors='black', linewidths=1.5, zorder=3)
 
        # ── Équipe 2 (rouge) ─────────────────────────────────────────
        for j in range(len(traj2["x"])):
            if i < len(traj2["x"][j]):
                plt.plot(traj2["x"][j][:i+1], traj2["y"][j][:i+1],
                         color='tomato', alpha=0.25, lw=1.5)
                plt.scatter(traj2["x"][j][i], traj2["y"][j][i],
                            color='tomato', s=200,
                            edgecolors='black', linewidths=1.5, zorder=3)
        
        # ── Balle (jaune) ─────────────────────────────────────────      
        if i < len(traj_balle["x"]):
            plt.scatter(traj_balle["x"][i], traj_balle["y"][i],
                       color='gold', s=130,
                       edgecolors='black', linewidths=1.5,
                       zorder=5, marker='o')

        plt.legend(handles=legende, loc='upper left', fontsize=9)
        plt.title(f"Rugby RL OOP — step {i}")
        plt.pause(0.01)
    plt.show()

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
    t1, t2, tb, dist = match(net1, net2, traces1, traces2)

    historique.append(dist)

    end = time.time()
    print(f"Époque {epoque:3d} | Distance J0 = {dist:.2f} | {end - start:.1f}s")
    start = end

    if epoque % 5 == 0:
        print(f"  → Animation époque {epoque}")
        afficher_match(t1, t2, tb)

    epsilon = max(0.01, epsilon * 0.995)

plt.figure()
plt.plot(historique)
plt.title("Progression de l'IA (Version Objet) — Distance finale")
plt.xlabel("Époque")
plt.ylabel("Position X")
plt.grid(True)
plt.tight_layout()
plt.show()