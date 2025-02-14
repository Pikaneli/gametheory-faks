import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, agent_id, true_value, noise_std=5.0):
        self.id = agent_id
        self.state = true_value + np.random.normal(0, noise_std)
        self.neighbors = []
    
    def update_state(self, weights, neighbor_states):
        consensus_update = sum(weights[i] * (neighbor_states[i] - self.state) for i in range(len(neighbor_states)))
        self.state += consensus_update
    
    def __repr__(self):
        return f"Agent(id={self.id}, state={self.state:.2f})"

class Broker:
    def __init__(self, num_agents, true_value, noise_std=5.0):
        self.num_agents = num_agents
        self.true_value = true_value
        self.agents = [Agent(i, true_value, noise_std) for i in range(num_agents)]
        self.graph = self._generate_communication_graph()
        self.conn_matrix = nx.to_numpy_array(self.graph)
        self.weights = self._compute_metropolis_weights()
    
    def _generate_communication_graph(self):
        graph = nx.erdos_renyi_graph(self.num_agents, 0.4, seed=42)
        while not nx.is_connected(graph):
            graph = nx.erdos_renyi_graph(self.num_agents, 0.4, seed=42)
        return graph
    
    def _compute_metropolis_weights(self):
        adj_matrix = self.conn_matrix
        degrees = adj_matrix.sum(axis=1)
        weights = np.zeros_like(adj_matrix)
        
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if adj_matrix[i, j] == 1:  # If there's a connection
                    weights[i, j] = 1 / max(degrees[i], degrees[j])
            weights[i, i] = 1 - weights[i].sum()  # Self-weight to ensure sum is 1
        
        return weights
    
    def get_neighbor_states(self, agent_id):
        neighbors = list(self.graph.neighbors(agent_id))
        return [self.agents[neighbor].state for neighbor in neighbors], neighbors
    
    def check_convergence(self, tolerance=1e-3):
        states = [agent.state for agent in self.agents]
        return np.max(np.abs(np.array(states) - self.true_value)) < tolerance

class WeightedAveragingConsensus:
    def __init__(self, broker):
        self.broker = broker
        self.iteration = 0
        self.history = []
    
    def run(self, max_iterations=1000, tolerance=1e-3):
        while self.iteration < max_iterations:
            self.history.append([agent.state for agent in self.broker.agents])
            
            for agent in self.broker.agents:
                neighbor_states, neighbors = self.broker.get_neighbor_states(agent.id)
                weights = [self.broker.weights[agent.id, neighbor] for neighbor in neighbors]
                agent.update_state(weights, neighbor_states)
            
            self.iteration += 1
            if self.broker.check_convergence(tolerance):
                print(f"Consensus reached at iteration {self.iteration}")
                break
    
    def visualize(self):
        self.history = np.array(self.history)
        plt.figure(figsize=(10, 6))
        plt.xlim([0, 50])
        
        for agent_id in range(self.broker.num_agents):
            plt.plot(self.history[:, agent_id], label=f"Agent {agent_id}")
        
        plt.axhline(self.broker.true_value, color='red', linestyle='--', label="True Value")
        plt.title("Weighted Averaging Consensus Protocol")
        plt.xlabel("Iteration")
        plt.ylabel("Agent State")
        plt.legend()
        plt.show()

# Simulation
num_agents = 10
true_value = 56
noise_std = 7.0

broker = Broker(num_agents, true_value, noise_std)
protocol = WeightedAveragingConsensus(broker)
protocol.run(max_iterations=1000, tolerance=1e-3)
protocol.visualize()
