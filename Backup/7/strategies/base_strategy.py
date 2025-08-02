class BaseStrategy:
    """
    Classe base para todas as estratégias.
    Todas as estratégias devem implementar generate_signals().
    """
    def __init__(self, name="Base Strategy"):
        self.name = name

    def generate_signals(self, data):
        raise NotImplementedError("Subclasses devem implementar generate_signals()")

