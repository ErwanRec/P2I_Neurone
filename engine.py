import math

class Value:
    """
    Stocke un scalaire et son gradient.
    Basé sur micrograd (Karpathy), avec ajout de sigmoid().
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ------------------------------------------------------------------ #
    #  Opérations de base                                                  #
    # ------------------------------------------------------------------ #

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Seules les puissances int/float sont supportées"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    #  Fonctions d'activation                                              #
    # ------------------------------------------------------------------ #

    def relu(self):
        """ReLU — utilisée sur les couches cachées."""
        out = Value(max(0.0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        """
        Sigmoïde — utilisée sur la couche de sortie.
        Clippe l'entrée pour éviter l'overflow numérique.
        """
        x = max(-500.0, min(500.0, self.data))
        s = 1.0 / (1.0 + math.exp(-x))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1.0 - s) * out.grad
        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    #  Rétropropagation                                                    #
    # ------------------------------------------------------------------ #

    def backward(self):
        """Lance la rétropropagation sur tout le graphe de calcul."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # ------------------------------------------------------------------ #
    #  Surcharges Python                                                   #
    # ------------------------------------------------------------------ #

    def __neg__(self):              return self * -1
    def __radd__(self, other):      return self + other
    def __sub__(self, other):       return self + (-other)
    def __rsub__(self, other):      return other + (-self)
    def __rmul__(self, other):      return self * other
    def __truediv__(self, other):   return self * other ** -1
    def __rtruediv__(self, other):  return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
