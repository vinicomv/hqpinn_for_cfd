import numpy as np
import matplotlib.pyplot as plt
import torch as tc
import torch.nn as nn
import pennylane as qml
from pennylane.qnn.torch import TorchLayer

class PINN(nn.Module):
    def __init__(self, neurons=5, depth=2, activation=nn.Tanh(), outdim=1):
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(2, neurons)])
        self.fcs.extend(nn.Linear(neurons, neurons) for _ in range(depth-1))
        self.out = nn.Linear(neurons, outdim)
        self.act = activation
    def forward(self, xt):
        z = xt
        for fc in self.fcs: z = self.act(fc(z))
        return self.out(z)
    

class HPINN(nn.Module):
    # Classical pre -> quantum -> classical post to scalar.
    def __init__(self, n_qubits=2, q_layers=1, pre_neurons=5, pre_depth=2, activation=nn.Tanh()):
        super().__init__()
        pre = [nn.Linear(2, pre_neurons), activation]
        for _ in range(pre_depth-1): pre += [nn.Linear(pre_neurons, pre_neurons), activation]
        pre += [nn.Linear(pre_neurons, n_qubits)]
        self.pre = nn.Sequential(*pre)
        dev = qml.device("default.qubit", wires=n_qubits)
        
    # Defining variational ansatz
        def cascade_ansatz(weights, wires):
            n_qubits = len(wires)
            L = 0
            for l in range(q_layers):
                for i in range(n_qubits):
                    qml.RX(weights[l, i, 0], wires=wires[i])
                    qml.RZ(weights[l, i, 1], wires=wires[i])
                for i in range(n_qubits):
                    control = wires[i-1]
                    target = wires[i]
                    qml.ctrl(qml.RX, control=control)(weights[l, i, 2], wires=target)
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))

            #Pennylane comes built-in with many ansatzes, uncomment to try them out.
            #qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            #qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

            cascade_ansatz(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit 
        self.weight_shape = (q_layers, n_qubits, 3)
        self.quantum = TorchLayer(circuit, {"weights": self.weight_shape})
        self.post = nn.Linear(n_qubits, pre_neurons)
        self.final = nn.Linear(pre_neurons, 1)

    def forward(self, xt):
        pre_out = self.pre(xt)
        q_out = self.quantum(pre_out)
        post_out = tc.tanh(self.post(q_out)) 
        return self.final(post_out)
    # prints circuit if requested
    def show_circuit(self, sample_input=None):
        if sample_input is None:
            sample_input = tc.zeros(self.pre[-1].out_features)
        q_layers, n_qubits, _ = self.weight_shape
        sample_weights = tc.zeros((q_layers, n_qubits, 3))
        return qml.draw_mpl(self.circuit)(sample_input, sample_weights)

