import random
from engine import Value


class Module:
    """Classe de base pour tous les modules du réseau."""

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


# ------------------------------------------------------------------ #
#  Neurone                                                             #
# ------------------------------------------------------------------ #

class Neuron(Module):
    """
    Un neurone avec une activation configurable.
      - 'relu'    : couches cachées
      - 'sigmoid' : couche de sortie
      - 'linear'  : pas d'activation (utile pour déboguer)
    """

    def __init__(self, nin, activation='relu'):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        assert activation in ('relu', 'sigmoid', 'linear'), \
            f"Activation inconnue : {activation}"
        self.activation = activation

    def __call__(self, x):
        # Produit scalaire w·x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == 'relu':
            return act.relu()
        elif self.activation == 'sigmoid':
            return act.sigmoid()
        else:  # linear
            return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron(activation={self.activation}, nin={len(self.w)})"


# ------------------------------------------------------------------ #
#  Couche                                                              #
# ------------------------------------------------------------------ #

class Layer(Module):
    """
    Une couche dense de `nout` neurones, chacun recevant `nin` entrées.
    """

    def __init__(self, nin, nout, activation='relu'):
        self.neurons = [Neuron(nin, activation=activation) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        # Si une seule sortie, on renvoie le scalaire directement
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer([{', '.join(str(n) for n in self.neurons)}])"


# ------------------------------------------------------------------ #
#  Réseau multi-couches (MLP)                                          #
# ------------------------------------------------------------------ #

class MLP(Module):
    """
    Réseau de neurones entièrement connecté.

    Architecture :
      - Couches cachées : activation ReLU
      - Couche de sortie : activation Sigmoïde

    Paramètres
    ----------
    nin   : int         – taille de l'entrée
    nouts : list[int]   – taille de chaque couche (la dernière = sortie)

    Exemple
    -------
    >>> net = MLP(48, [16, 16, 16, 1])   # 48 entrées, 3 couches cachées, 1 sortie
    """

    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            is_last = (i == len(nouts) - 1)
            activation = 'sigmoid' if is_last else 'relu'
            self.layers.append(Layer(sizes[i], sizes[i + 1], activation=activation))

    def __call__(self, x):
        # x peut être une liste de float ou de Value
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def scalar_output(self, x):
        """Renvoie la sortie sous forme de float Python (utile pour TD-Lambda)."""
        out = self(x)
        if isinstance(out, Value):
            return out.data
        # Si la couche de sortie a plusieurs neurones
        return [v.data for v in out]

    def __repr__(self):
        return f"MLP([{', '.join(str(l) for l in self.layers)}])"
