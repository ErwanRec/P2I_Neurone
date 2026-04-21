import numpy as np
import fonction_RL as RL
import matplotlib.pyplot as plt
import copy
import random
import json
from nn import MLP


global caracteristique , mb , taille_terrain, g, T , t , n , K , actions_sans_balle , action_avec_balle_1 , action_avec_balle_2 , taille_liste_action
K = 0.6
n = 4 # le nombre de joueur total
T = 60 # temps final
J = []
g = 9.81
caracteristique = [{"m": 91 , "r": 1/6},{"m": 80 , "r": 1/6},{"m": 75 , "r": 1/6},{"m": 83 , "r": 1/6},{"m": 83 , "r": 1/6},{"m": 100 , "r": 1/6},{"m": 86 , "r": 1/6},{"m": 76 , "r": 1/6}] # massse en kg , r en m
mb = 0.410
taille_terrain = {"longeur": 100 , "largeur": 50}

actions_sans_balle = []
action_avec_balle_1 = []
action_avec_balle_2 = []

action_joueurs = []
for i in [-1, 0, 1]:  # Au lieu de range(3)
    for j in [-1, 0, 1]:
        Fx = i * 50 
        Fy = j * 50
        action_joueurs.append([Fx, Fy, -1])
        
action_joueur_balle = []
for i in [-1, 0, 1]:  # Au lieu de range(3)
  for j in [-1, 0, 1]:
    Fx = i * 50 
    Fy = j * 50
    action_joueur_balle.append([Fx,Fy,-1])

    for cible_id in range(n):
        action_joueur_balle.append([Fx, Fy, cible_id])

for a in action_joueurs:
    for b in action_joueurs:
        actions_sans_balle.append([a,b])

for a in action_joueur_balle:
    for b in action_joueurs:
        action_avec_balle_1.append([a,b])

for a in action_joueurs:
    for b in action_joueur_balle:
        action_avec_balle_2.append([a,b])


taille_liste_action = len(action_avec_balle_1)

# -----------------------------------------------------------------------------------------------------------------

def decision_avec_probabilite(p , J):
    f = J.fatigue
    r = np.random.uniform(0, 1)
    return (r> (p + (f*10**(-2))) , r)

def dico():
    filin = open("activations.txt", "r")
    lignes = filin.readlines()
    n = 0
    for ligne in lignes:
        n += 1
        if ligne[0] == "{" and n > 1:
            break
    text1 = ""
    for i in range(n - 1):
        text1 = text1 + lignes[i]

    text2 = ""
    for j in range(n-1,len(lignes)):
        text2 = text2 + lignes[j]
    # Préparer un dictionnaire pour servir de contexte
    context = {}
    # Transformer la chaîne en remplaçant 'array' par 'np.array'
    text1 = text1.replace("array", "np.array")
    # Utiliser exec pour évaluer la chaîne dans le contexte donné
    exec(f"result = {text1}", {'np': np}, context)
    # Extraire le résultat
    A1 = context['result']

    # Préparer un dictionnaire pour servir de contexte
    context = {}
    # Transformer la chaîne en remplaçant 'array' par 'np.array'
    text2 = text2.replace("array", "np.array")
    # Utiliser exec pour évaluer la chaîne dans le contexte donné
    exec(f"result = {text2}", {'np': np}, context)
    # Extraire le résultat
    A2 = context['result']

    return A1 , A2

def angle(x1,y1,x2,y2):
  deltax = x2 - x1
  deltay = y2 - y1
  if deltax == 0:
    if deltay == 0:
      return 0
    else:
      return np.pi/2
  else:
    return np.arctan(deltay/deltax)

def intervale_2proba(a,b):
    fst = np.random.uniform(a,b)
    snd = np.random.uniform(a, fst)
    return fst, snd

def DeltaEC (vitesse1 , m1, vitesse2, m2, co1 , co2 ):
    DEC = (1/2)*m1*(vitesse1*co1)**2 + (1/2)*m2*(vitesse2*co2)**2
    return DEC

def circle_to_polygon(center_x, center_y, radius, theta,n=1, num_points=30):
    """Convertit un cercle en un polygone avec un nombre donné de points."""
    angles = [((n*i * np.pi) / num_points) - theta  for i in range(num_points)]
    points = [(center_x + radius * np.cos(angle), center_y + radius * np.sin(angle)) for angle in angles]
    return points

def d(X,Y):
  return np.sqrt((X[0] - Y[0])**2 + (X[1] - Y[1])**2)

def list_vide(n):
  C = []
  for i in range(n):
    C.append(0)
  return C

def dans(J,l):
  if l == []:
    return False
  for i in l:
    for j in i:
      if j == "pos" or j == "vit" or j == "acc":
        for k in J[j]:
          if not(np.array_equal(J[j][k],i[j][k])):
            return False
      else:
        if not(J[j] == i[j]):
          return False
  return True

def _extraire_action_oop(act):
    """
    Déplie une action quelle que soit son niveau d'imbrication.
 
    Cas supportés :
      [Fx, Fy, passe]                         → forme normale
      [[Fx, Fy, passe], [Fx, Fy, passe]]      → paire d'actions (on prend la 1ère)
    """
    while isinstance(act, list) and len(act) > 0 and isinstance(act[0], list):
        act = act[0]
 
    if not isinstance(act, list) or len(act) < 3:
        return 0.0, 0.0, 0  # immobile par défaut
 
    Fx, Fy, passe = act[0], act[1], act[2]
    if isinstance(Fx, list): Fx = 0.0
    if isinstance(Fy, list): Fy = 0.0
    return float(Fx), float(Fy), passe

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

def match(net1: MLP, net2: MLP, traces1: dict, traces2: dict, PARAMS_MATCH, epsilon):
    env = Match(PARAMS_MATCH)
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
            env.balle.vol_libre(t_step, g)
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
        plt.xlim(0, taille_terrain["longeur"])
        plt.ylim(0, taille_terrain["largeur"])
        plt.axvline(x=taille_terrain["longeur"], color='green', lw=5, alpha=0.5)
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

# -----------------------------------------------------------------------------------------------------------------

class Balle:
    def __init__(self, T_max):
        self.T_max = T_max
        self.x = np.zeros(T_max)
        self.y = np.zeros(T_max)
        self.z = np.zeros(T_max)
        self.vx = np.zeros(T_max)
        self.vy = np.zeros(T_max)
        self.porteur = -1  # -1 = balle au sol/en l'air, sinon ID du joueur
        self.dernier_passeur = -1
    
    def vol_libre(self, t, g):
        """Fait continuer la trajectoire de la balle quand elle n'a plus de porteur."""
        # Elle continue sur sa vitesse (Inertie)
        self.x[t] = self.x[t-1] + self.vx[t-1]
        self.y[t] = self.y[t-1] + self.vy[t-1]
        self.z[t] = max(0, self.z[t-1] - (0.5 * g)) # Elle tombe, mais s'arrête au sol (z=0)
        
        # Frottement de l'air/sol
        self.vx[t] = self.vx[t-1] * 0.95
        self.vy[t] = self.vy[t-1] * 0.95
        
    def suivre_joueur(self, joueur, t):
        """Si un joueur porte la balle, elle prend ses coordonnées."""
        self.x[t] = joueur.pos_x[t]
        self.y[t] = joueur.pos_y[t]
        self.z[t] = 1.0 # Hauteur des mains environ
        self.vx[t] = joueur.vit_x[t]
        self.vy[t] = joueur.vit_y[t]
        
class Joueur:
    def __init__(self, id_joueur, equipe, masse, rayon, x0, y0, T_max):
        self.id = id_joueur
        self.equipe = equipe
        self.m = masse
        self.r = rayon
        
        # Vecteurs d'état (historique)
        self.pos_x = np.zeros(T_max)
        self.pos_y = np.zeros(T_max)
        self.vit_x = np.zeros(T_max)
        self.vit_y = np.zeros(T_max)
        self.acc_x = np.zeros(T_max)
        self.acc_y = np.zeros(T_max)
        
        # Initialisation à t=0
        self.pos_x[0], self.pos_y[0] = x0, y0
        
        # États
        self.fatigue = 0
        self.immobilise_timer = 0 
        self.statut = 0  # 0: normal, 1: au sol (plaqué), 4: dans un ruck
        
    def get_pos(self, t):
        return self.pos_x[t], self.pos_y[t]
    
    def appliquer_mouvement(self, Fx, Fy, t, K):
        # Fx = [Fx, Fy, [F, theta, phi]]
        # Fy = [Fx, Fy, [0,0,0]]
        A = np.sqrt(Fx**2 + Fy**2) / self.m
        
        # Mise à jour de la vitesse (formule avec frottement K)
        # Note : On utilise un simple delta t = 1 pour la simulation
        self.vit_x[t] = A * np.exp(-K/self.m) * np.cos(np.arctan2(Fy, Fx)) if Fx != 0 or Fy != 0 else 0
        self.vit_y[t] = A * np.exp(-K/self.m) * np.sin(np.arctan2(Fy, Fx)) if Fx != 0 or Fy != 0 else 0
        
        self.acc_x[t] = Fx / self.m
        self.acc_y[t] = Fy / self.m
        
        # Mise à jour de la position
        self.pos_x[t] = self.pos_x[t-1] + self.vit_x[t]
        self.pos_y[t] = self.pos_y[t-1] + self.vit_y[t]

        v = np.sqrt(self.vit_x[t]**2 + self.vit_y[t]**2)
        self.fatigue += 0.01 if v > 3 else -0.01 # Le joueur se fatigue en bougeant
    
class Terrain:
    def __init__(self, longueur=100, largeur=70, nb_joueurs=4):
        self.longueur = longueur
        self.largeur = largeur
        self.t = 0 # Le pas de temps actuel
        self.joueurs = []
        self.balle = Balle(longueur/2, largeur/2)
        
        # Initialisation automatique des joueurs
        for i in range(nb_joueurs):
            eq = 1 if i < nb_joueurs/2 else 2
            x_init = 10 if eq == 1 else longueur - 10
            self.joueurs.append(Joueur(i, eq, x_init, largeur/2 + (i%2)*5))

    def reinit(self):
        self.t = 0
        # Réinitialiser positions des joueurs et balle...  a faire 
        
class Match:
    def __init__(self, params):
        self.T_max = params['T']
        self.K = params['K']
        self.g = params['g']
        self.mb = params['mb']
        self.longeur = params['taille_terrain']['longeur']
        self.largeur = params['taille_terrain']['largeur']
        
        self.t = 0
        self.joueurs = []
        self.balle = Balle(self.T_max)
        
        self.ruck_coor = [-1, -1]
        self.timer_ruck = 0
        self.score      = {1: 0, 2: 0}
        
        # Création des joueurs
        caracteristiques = params['caracteristique']
        n = params['n']
        for i in range(n):
            eq = 1 if i < n/2 else 2
            if 'positions_initiales' in params:
                x_init, y_init = params['positions_initiales'][i]
            else:
                x_init    = self.longeur * 0.45 if eq == 1 else self.longeur * 0.55
                y_init   = self.largeur / 2 + (i % 2) * 10 - 5
            masse = caracteristiques[i]["m"]
            rayon = caracteristiques[i]["r"]
            self.joueurs.append(Joueur(i, eq, masse, rayon, x_init, y_init, self.T_max))
        
        self.balle.porteur = 0
        self.balle.x[0] = self.joueurs[0].pos_x[0]
        self.balle.y[0] = self.joueurs[0].pos_y[0]
        self.balle.z[0] = 1.0
    
    def donner_balle_a(self, id_joueur):
        """Méthode propre pour forcer la balle dans les mains d'un joueur."""
        self.balle.porteur = id_joueur
        joueur = self.joueurs[id_joueur]
        self.balle.x[self.t] = joueur.pos_x[self.t]
        self.balle.y[self.t] = joueur.pos_y[self.t]
        self.balle.z[self.t] = 1.0
          
    def reset(self):
        """Relance le match (remplace relance_match)."""
        self.__init__({
            'T': self.T_max, 'K': self.K, 'g': self.g, 'mb': self.mb,
            'taille_terrain': {'longeur': self.longeur, 'largeur': self.largeur},
            'caracteristique': caracteristique, 'n': len(self.joueurs)
        })
    
    def clone(self):
        """Copie profonde pour la simulation IA."""
        return copy.deepcopy(self)

    def contact(self, j1, j2):
        """Vérifie si deux joueurs sont en contact au temps actuel t."""
        dx = j1.pos_x[self.t] - j2.pos_x[self.t]
        dy = j1.pos_y[self.t] - j2.pos_y[self.t]
        return np.sqrt(dx**2 + dy**2) < (j1.r + j2.r)

    def plaquage(self, attaquant, defenseur):
        """Gère la logique d'un plaquage."""
        print(f"Plaquage entre Joueur {attaquant.id} et Joueur {defenseur.id} !")
        # Immobilisation des joueurs
        attaquant.statut = 1
        defenseur.statut = 1
        attaquant.vit_x[self.t] = 0
        attaquant.vit_y[self.t] = 0
        defenseur.vit_x[self.t] = 0
        defenseur.vit_y[self.t] = 0
        defenseur.fatigue  += 1.0

        # Déclenchement d'un ruck
        self.timer_ruck = 4 # Durée du ruck
        self.ruck_coor = [attaquant.pos_x[self.t], attaquant.pos_y[self.t]]

    def step(self, actions_eq1, actions_eq2):
        """Remplace action_match. Avance la simulation d'un pas de temps (t -> t+1)."""
        if self.etat_final():
            return True # Fin du match
            
        self.t += 1
        t = self.t
        nb = len(self.joueurs) // 2
        actions_totales = list(actions_eq1)[:nb] + list(actions_eq2)[:nb]
        
        # Appliquer les mouvements
        for i, joueur in enumerate(self.joueurs):
            Fx, Fy, _ = _extraire_action_oop(actions_totales[i])
            if joueur.statut == 0: # S'il est libre de bouger
                joueur.appliquer_mouvement(Fx, Fy, t, self.K)
                if joueur.pos_x[t] < -1:
                    joueur.pos_x[t] = 0
                    joueur.vit_x[t] = 0 # Coupe l'élan contre le bord
                elif joueur.pos_x[t] > self.longeur+1:
                    joueur.pos_x[t] = self.longeur
                    joueur.vit_x[t] = 0
                    
                # On bloque en Y (largeur)
                if joueur.pos_y[t] < -1:
                    joueur.pos_y[t] = 0
                    joueur.vit_y[t] = 0
                elif joueur.pos_y[t] > self.largeur+1:
                    joueur.pos_y[t] = self.largeur
                    joueur.vit_y[t] = 0
            else:
                # S'il est au sol ou dans un ruck, il reste sur place
                joueur.pos_x[t] = joueur.pos_x[t-1]
                joueur.pos_y[t] = joueur.pos_y[t-1]
                joueur.vit_x[t] = 0
                joueur.vit_y[t] = 0
                if joueur.statut > 0:
                    joueur.statut -= 1 # décompte du timer d'immobilisation
        # Gestion de la balle
        porteur_id = self.balle.porteur
        if porteur_id != -1:
            porteur = self.joueurs[porteur_id]
            _, _, cible_id = _extraire_action_oop(actions_totales[porteur_id])

            if cible_id != -1 and cible_id != porteur_id and self.joueurs[cible_id].equipe == porteur.equipe:
                cible = self.joueurs[cible_id]
                est_en_arriere = False
                
                # Vérification de la passe en arrière
                if porteur.equipe == 1 and cible.pos_x[t-1] < porteur.pos_x[t-1]:
                    est_en_arriere = True
                elif porteur.equipe == 2 and cible.pos_x[t-1] > porteur.pos_x[t-1]:
                    est_en_arriere = True
                    
                if est_en_arriere:
                    # Passe directe instantanée dans les mains !
                    self.balle.porteur = cible_id
                    self.balle.dernier_passeur = porteur_id
                    self.balle.suivre_joueur(cible, t)
                else:
                    self.balle.suivre_joueur(porteur, t)
            else:
                self.balle.suivre_joueur(porteur, t)
                
        # Contrainte stricte : les attaquants restent à 3 mètres max (axe X) du porteur
        porteur_id_actuel = self.balle.porteur # On actualise au cas où il y a eu une passe
        if porteur_id_actuel != -1:
            porteur_actuel = self.joueurs[porteur_id_actuel]
            for j in self.joueurs:
                if j.equipe == porteur_actuel.equipe and j.id != porteur_id_actuel:
                    ecart_x = j.pos_x[t] - porteur_actuel.pos_x[t]
                    # Si le joueur est à plus de 3m devant ou derrière, on le bloque à 3m
                    if ecart_x > 3.0:
                        j.pos_x[t] = porteur_actuel.pos_x[t] + 3.0
                    elif ecart_x < -3.0:
                        j.pos_x[t] = porteur_actuel.pos_x[t] - 3.0
                        
        # Vérification des collisions (Plaquages)
        for i in range(len(self.joueurs)):
            for j in range(i + 1, len(self.joueurs)):
                j1, j2 = self.joueurs[i], self.joueurs[j]
                if j1.equipe != j2.equipe and self.contact(j1, j2):
                    # Si l'un des deux a la balle, c'est un plaquage
                    if self.balle.porteur == j1.id:
                        self.plaquage(j1, j2)
                    elif self.balle.porteur == j2.id:
                        self.plaquage(j2, j1)

        return self.etat_final()

    def etat_final(self):
        """Vérifie si le match est terminé (temps écoulé, essai, touche)."""
        if self.t >= self.T_max - 1:
            return True
            
        porteur_id = self.balle.porteur
        if porteur_id != -1:
            porteur = self.joueurs[porteur_id]
            # Vérification de l'essai
            if porteur.equipe == 1 and porteur.pos_x[self.t] >= self.longeur:
                self.score[1] += 5
                print("ESSAI EQUIPE 1 ! (t={self.t})")
                return True
            elif porteur.equipe == 2 and porteur.pos_x[self.t] <= 0:
                self.score[2] += 5
                print("ESSAI EQUIPE 2 ! (t={self.t})")
                return True
                
            # Vérification de la touche
            if porteur.pos_y[self.t] < 0 or porteur.pos_y[self.t] > self.largeur:
                print("TOUCHE ! (t={self.t})")
                if porteur_id < int(n/2):
                    self.donner_balle_a(int(n/2))
                    self.joueurs[int(n/2)].pos_y[self.t] = porteur.pos_y[self.t-1]
                    self.joueurs[int(n/2)].pos_x[self.t] = porteur.pos_x[self.t-1]
                    self.joueurs[int(n/2)].vit_x[self.t] = 0
                    self.joueurs[int(n/2)].vit_y[self.t] = 0
                    self.joueurs[int(n/2)].acc_x[self.t] = 0
                    self.joueurs[int(n/2)].acc_y[self.t] = 0
                    porteur.pos_y[self.t] = porteur.pos_y[self.t-1]
                    porteur.pos_x[self.t] = porteur.pos_x[self.t-1] -5
                    porteur.vit_x[self.t] = 0
                    porteur.vit_y[self.t] = 0
                    porteur.acc_x[self.t] = 0
                    porteur.acc_y[self.t] = 0
                    
                else :
                    self.donner_balle_a(int(n/2)-2)
                    self.joueurs[int(n/2)-2].pos_y[self.t] = porteur.pos_y[self.t-1]
                    self.joueurs[int(n/2)-2].pos_x[self.t] = porteur.pos_x[self.t-1]
                    self.joueurs[int(n/2)].vit_x[self.t] = 0
                    self.joueurs[int(n/2)].vit_y[self.t] = 0
                    self.joueurs[int(n/2)].acc_x[self.t] = 0
                    self.joueurs[int(n/2)].acc_y[self.t] = 0
                    porteur.pos_y[self.t] = porteur.pos_y[self.t-1]
                    porteur.pos_x[self.t] = porteur.pos_x[self.t-1] +5
                    porteur.vit_x[self.t] = 0
                    porteur.vit_y[self.t] = 0
                    porteur.acc_x[self.t] = 0
                    porteur.acc_y[self.t] = 0
        return False
