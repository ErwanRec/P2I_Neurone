import numpy as np
import document as dc
import fonction_RL as RL
from nn import MLP

# Recréer l'architecture vide
net1_test = MLP(48, [16, 16, 16, 1])
net2_test = MLP(48, [16, 16, 16, 1])

# Charger les cerveaux entrainés
dc.charger_reseau(net1_test, "ia_equipe1_entrainee.json")
dc.charger_reseau(net2_test, "ia_equipe2_entrainee.json")

positions_initiales = {
    0: (40, 25), 1: (40, 30), # Equipe 1
    2: (60, 25), 3: (60, 30)  # Equipe 2
}

# Créer une NOUVELLE initialisation de terrain
PARAMS_MATCH = {
    'T': dc.T, 'K': dc.K, 'g': dc.g, 'mb': dc.mb, 'n': dc.n,
    'taille_terrain': dc.taille_terrain,
    'caracteristique': dc.caracteristique,
    'positions_initiales': positions_initiales
} 
env_test = dc.Match(PARAMS_MATCH)
env_test.donner_balle_a(0)

for joueur in env_test.joueurs:
     joueur.pos_x[0] += np.random.uniform(-3, 3)
     joueur.pos_y[0] += np.random.uniform(-3, 3)

nb = int(dc.n / 2)
zero = [0.0, 0.0, -1]
traj_balle = {"x": [], "y": [], "porteur": []}

# Jouer le match SANS ENTRAINEMENT (epsilon = 0)
while not env_test.etat_final():
    t_step = env_test.t
    S = RL.recuperer_etat_au_temps_t(env_test, t_step)
    
    pid = env_test.balle.porteur
    traj_balle["porteur"].append(pid)
    if pid != -1:
        traj_balle["x"].append(env_test.joueurs[pid].pos_x[t_step])
        traj_balle["y"].append(env_test.joueurs[pid].pos_y[t_step])
    else:
        traj_balle["x"].append(env_test.balle.x[t_step])
        traj_balle["y"].append(env_test.balle.y[t_step])

    # Toujours la meilleure action, pas de random()
    action1 = dc.choisir_meilleure_action(env_test, S, net1_test, equipe=1)
    action2 = dc.choisir_meilleure_action(env_test, S, net2_test, equipe=2)

    # Création des listes d'actions et step
    a1_liste = [action1] + [zero] * (nb - 1)
    a2_liste = [action2] + [zero] * (nb - 1)
    env_test.step(a1_liste, a2_liste)

# Afficher le résultat
end_t = env_test.t + 1
traj1 = {"x": [], "y": []}
traj2 = {"x": [], "y": []}

for i in range(nb):
    traj1["x"].append(env_test.joueurs[i].pos_x[:end_t])
    traj1["y"].append(env_test.joueurs[i].pos_y[:end_t])
    traj2["x"].append(env_test.joueurs[nb+i].pos_x[:end_t])
    traj2["y"].append(env_test.joueurs[nb+i].pos_y[:end_t])

dc.afficher_match(traj1, traj2, traj_balle)