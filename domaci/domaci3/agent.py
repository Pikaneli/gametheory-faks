import numpy as np
"""Komentar2"""
class Agent:
    def __init__(self, agent_id, true_value, noise_std=5.0):
        """
        Inicijalizuje agenta sa:
        - agent_id: ID agenta
        - true_value: stvarna vrednost koja biva estimirana
        - noise_std: sum u vidu standarne devijacije
        - stanje: Inicijalno merenje koje na stvarnu vrednost koju meri senzor dodaje neki sum
        - neighbors`: skup suseda za svakog agenta
        """
        self.id = agent_id
        self.state = true_value + np.random.normal(0, noise_std)  # pocetni sum merenja
        self.neighbors = []  # susedni agenti
    
    def update_state(self, weights, neighbor_states, true_value, alpha=0.01):
        """
        Azurira agentsko stanje postujuci protokol.

        - weights: faktor za kontrolu konvergiranja i prihvatanja razlike u odnosu na susede
        - neighbor_states: stanja susednih agenata
        - alpha: bias faktor prema stvarnoj vrednosti (0.01)

        - Agentovo stanje tezi da se izjednaci ili bar nadje na pola sa susedima uz pomoc tezinskog faktora
        """
        consensus_update = sum(weights[i] * (neighbor_states[i] - self.state) for i in range(len(neighbor_states)))
        bias_update = alpha * (true_value - self.state)  # Add bias toward true value
        self.state += consensus_update + bias_update

    def __repr__(self):
        """
        Ispis i privera stanja, za debugging
        """
        return f"Agent(id={self.id}, state={self.state:.2f})"
