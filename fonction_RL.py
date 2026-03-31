import numpy as np
from engine import Value
from nn import MLP
import document as dc

# ------------------------------------------------------------------ #
#  Hyperparamètres                                                     #
# ------------------------------------------------------------------ #
Alpha  = 0.01   # taux d'apprentissage
Lambda = 0.9    # decay traces d'éligibilité
Gamma  = 0.99   # facteur d'actualisation

# ------------------------------------------------------------------ #
#  Interface réseau ↔ numpy                                            #
# ------------------------------------------------------------------ #
def etat_vers_liste(S: np.ndarray) -> list:
    return [float(v) for v in S.flatten()]

def forward(net: MLP, S: np.ndarray) -> Value:
    out = net(etat_vers_liste(S))
    return out if isinstance(out, Value) else out[0]

def valeur(net: MLP, S: np.ndarray) -> float:
    return net.scalar_output(etat_vers_liste(S))

def choix_batch(net: MLP, etats: list) -> np.ndarray:
    return np.array([net.scalar_output(etat_vers_liste(S)) for S in etats], dtype=float)

# ------------------------------------------------------------------ #
#  Traces d'éligibilité                                                #
# ------------------------------------------------------------------ #
def init_traces(net: MLP) -> dict:
    return {id(p): 0.0 for p in net.parameters()}

def TD_lambda(net: MLP, traces: dict, S: np.ndarray, S_next: np.ndarray, R: float) -> float:
    net.zero_grad()
    v0_val = forward(net, S)
    v0_val.backward()

    v1 = valeur(net, S_next)
    delta = R + Gamma * v1 - v0_val.data

    for p in net.parameters():
        key = id(p)
        traces[key] = Gamma * Lambda * traces[key] + p.grad
        p.data     += Alpha * delta * traces[key]
        p.grad      = 0.0

    return delta

# ------------------------------------------------------------------ #
#  Récompense                                                          #
# ------------------------------------------------------------------ #
def calculer_recompense(env, final: bool) -> float:
    """Récompense basée sur la progression vers la ligne d'essai."""
    pos_x = env.balle.x[env.t]
    cible = env.longeur
    if final:
        # Bonus si un essai est marqué (le porteur est derrière la ligne)
        if env.balle.porteur != -1 and env.joueurs[env.balle.porteur].pos_x[env.t] >= cible:
            return 10.0
        return 0.0 # Si ça finit en touche ou fin du temps sans essai
    return (pos_x / cible) * 0.1

# ------------------------------------------------------------------ #
#  État (Adapté à l'Orienté Objet)                                     #
# ------------------------------------------------------------------ #
def recuperer_etat_au_temps_t(env, t: int) -> np.ndarray:
    """Vecteur d'état (48,) au temps t lu depuis les objets Joueur et Balle."""
    S = []
    for j in env.joueurs:
        B = 1.0 if env.balle.porteur == j.id else 0.0 # B = 1 si le joueur a la balle
        S += [j.pos_x[t], j.pos_y[t],
              j.vit_x[t], j.vit_y[t],
              j.acc_x[t], j.acc_y[t],
              float(j.statut), 0.0, float(j.fatigue), B] # 0.0 remplace l'ancienne variable "r"
              
    b = env.balle
    S += [b.x[t], b.y[t], b.z[t],
          b.vx[t], b.vy[t],
          0.0, 0.0, float(b.porteur)] # 0.0 pour l'accélération de la balle (non traquée)
    return np.array(S, dtype=float)


def etat(env, a1: list, a2: list) -> np.ndarray:
    """Simule une action sur une copie de l'environnement (MCTS-style)."""
    cop_env = env.clone() # Utilisation de la méthode OOP magique !
    cop_env.step(a1, a2)
    return recuperer_etat_au_temps_t(cop_env, cop_env.t)

# ------------------------------------------------------------------ #
#  Actions                                                             #
# ------------------------------------------------------------------ #
def actions(env, equipe: int, tout: bool = False) -> list:
    if tout:
        return dc.action_avec_balle_1
        
    porteur = env.balle.porteur
    if porteur == -1:
        return dc.actions_sans_balle
        
    equipe_porteur = env.joueurs[porteur].equipe
    if equipe == 1:
        return dc.action_avec_balle_1 if equipe_porteur == 1 else dc.action_avec_balle_2
    else:
        return dc.action_avec_balle_1 if equipe_porteur == 2 else dc.action_avec_balle_2