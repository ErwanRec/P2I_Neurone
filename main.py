import time
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import document as dc
import fonction_RL as RL
from nn import MLP

start = time.time()

# ------------------------------------------------------------------ #
#  Hyperparamètres                                                     #
# ------------------------------------------------------------------ #
epsilon = 0.8
NB_EPOQUES = 120

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
#  Sélection d'action                                                  #
# ------------------------------------------------------------------ #
def choisir_meilleure_action(env, S: np.ndarray, net, equipe: int):
    actions_possibles = RL.actions(env, equipe)
    nb = int(len(env.joueurs) / 2)
    zero = [0, 0, -1]
    
    LARGEUR_TERRAIN = 100 
    HAUTEUR_TERRAIN = 50  
    
    # On isole les joueurs de l'équipe concernée
    if equipe == 1:
        joueurs_equipe = env.joueurs[0:nb]
    else:
        joueurs_equipe = env.joueurs[nb:2*nb]

    # Cette liste va stocker les actions finales de chaque joueur
    actions_choisies = []

    # On boucle sur CHAQUE joueur de l'équipe
    for i, joueur_actif in enumerate(joueurs_equipe):
        
        # ÉLAGAGE : On filtre les actions possibles pour ce joueur précis
        actions_filtrees = []
        for act in actions_possibles:
            fx = act[0]
        if isinstance(fx, (list, np.ndarray)): 
            fx = fx[0]
            
        fy = act[1]
        if isinstance(fy, (list, np.ndarray)): 
            fy = fy[0]

        #  Récupération ultra-sécurisée des positions 
        px = joueur_actif.pos_x
        if isinstance(px, (list, np.ndarray)): 
            px = px[0]
            
        py = joueur_actif.pos_y  
        if isinstance(py, (list, np.ndarray)): 
            py = py[0]

        #  Ton élagage directionnel (pour avancer vers le but) 
        if equipe == 1 and fx < 0:
            continue
        if equipe == 2 and fx > 0:
            continue
            
        # Masquage pour ne pas sortir du terrain 
        if px <= 0 and fx < 0:
            continue
        if px >= LARGEUR_TERRAIN and fx > 0:
            continue
        if py <= 0 and fy < 0:
            continue
        if py >= HAUTEUR_TERRAIN and fy > 0:
            continue

        # Si l'action a survécu à tous ces tests, on la garde
        actions_filtrees.append(act)

        # Sécurité : s'il est bloqué de partout, il ne fait rien
        if not actions_filtrees:
            actions_filtrees = [zero]

        # ÉVALUATION : On teste les actions pour ce joueur
        etats_suivants = []
        for act in actions_filtrees:
            # Astuce : On prend les actions déjà choisies par les joueurs précédents, 
            # on ajoute l'action testée ('act') pour le joueur actuel, 
            # et on met 'zero' pour les joueurs de l'équipe qui n'ont pas encore joué.
            actions_test = actions_choisies + [act] + [zero] * (nb - 1 - i)
            
            if equipe == 1:
                a1 = actions_test
                a2 = [zero] * nb
            else:
                a1 = [zero] * nb
                a2 = actions_test
                
            etats_suivants.append(RL.etat(env, a1, a2))

        # CHOIX : On donne l'état au réseau et on prend la meilleure
        scores = RL.choix_batch(net, etats_suivants)
        meilleur_index = int(np.argmax(scores))
        
        # On verrouille l'action de ce joueur et on passe au suivant
        actions_choisies.append(actions_filtrees[meilleur_index])

    return actions_choisies

# ------------------------------------------------------------------ #
#  Un match complet (Boucle d'environnement)                           #
# ------------------------------------------------------------------ #
def match(net1: MLP, net2: MLP, traces1: dict, traces2: dict):
    env = dc.Match(PARAMS_MATCH)
    for joueur in env.joueurs:
        joueur.pos_x[0] += np.random.uniform(-5, 5)
        joueur.pos_y[0] += np.random.uniform(-5, 5)
    
    env.donner_balle_a(0) # Le joueur 0 commence avec la balle
    
    nb = int(len(env.joueurs) / 2)
    zero = [0, 0, -1]
    
    # Trajectoires joueurs
    traj1 = {"x": [[] for _ in range(nb)], "y": [[] for _ in range(nb)]}
    traj2 = {"x": [[] for _ in range(nb)], "y": [[] for _ in range(nb)]}
 
    # Trajectoire balle + porteur à chaque step
    traj_balle = {"x": [], "y": [], "porteur": []}

    while not env.etat_final():
        t_step = env.t
        S = RL.recuperer_etat_au_temps_t(env, t_step)

        pid = env.balle.porteur
        traj_balle["porteur"].append(pid)
        if pid != -1:
            traj_balle["x"].append(env.joueurs[pid].pos_x[t_step])
            traj_balle["y"].append(env.joueurs[pid].pos_y[t_step])
        else:
            env.balle.vol_libre(t_step, dc.g)
            traj_balle["x"].append(env.balle.x[t_step])
            traj_balle["y"].append(env.balle.y[t_step])
        
        # --- Choix actions ---
        if t_step == 0:
            action1 = [random.choice(RL.actions(env, 1)) for _ in range(nb)]
            action2 = [random.choice(RL.actions(env, 2)) for _ in range(nb)]
        else:
            if np.random.rand() < epsilon:
                action1 = [random.choice(RL.actions(env, 1)) for _ in range(nb)]
            else:
                action1 = choisir_meilleure_action(env, S, net1, equipe=1)

            if np.random.rand() < epsilon:
                action2 = [random.choice(RL.actions(env, 2)) for _ in range(nb)]
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

    # voir si utile

    # traj_balle["x"].append(bx)
    # traj_balle["y"].append(by)
    # traj_balle["porteur"].append(pid)
    
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
# Sauvegarde                                                           #
# ------------------------------------------------------------------ #

def sauvegarder_reseau(net, nom_fichier):
    """Sauvegarde les poids du réseau dans un fichier JSON."""
    # On extrait la valeur float de chaque objet Value
    poids = [p.data for p in net.parameters()]
    with open(nom_fichier, 'w') as f:
        json.dump(poids, f)
    print(f"Réseau sauvegardé dans {nom_fichier}")

def charger_reseau(net, nom_fichier):
    """Charge les poids d'un fichier JSON dans le réseau."""
    with open(nom_fichier, 'r') as f:
        poids = json.load(f)
    
    parametres = net.parameters()
    if len(poids) != len(parametres):
        print("Erreur : L'architecture du réseau ne correspond pas à la sauvegarde !")
        return
        
    for p, val in zip(parametres, poids):
        p.data = val # On injecte la valeur apprise
    print(f"Réseau chargé depuis {nom_fichier}")


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
