"""
Quantum Simulator for NLP applications.

This module provides quantum simulation capabilities using classical hardware
to emulate quantum-like behavior for NLP processing, including quantum circuits,
state evolution, and measurement operations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import math

# Import licensing
from ..licensing import validate_class_license, requires_license

# Optional PennyLane import for advanced quantum features
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    qml = None
    pnp = None
    PENNYLANE_AVAILABLE = False


class QuantumSimulator:
    """
    Classical quantum simulator for NLP applications.
    
    This simulator provides quantum-like computations using classical hardware,
    enabling experimentation with quantum-inspired NLP algorithms without
    requiring actual quantum hardware.
    
    Features:
    - Quantum circuit simulation with PennyLane
    - State vector representation and evolution
    - Quantum gate operations (Pauli, Hadamard, CNOT, etc.)
    - Measurement and expectation value computation
    - Entanglement and superposition modeling
    - Decoherence and noise simulation
    
    LICENSE REQUIRED: This class requires a valid license to operate.
    Contact bajpaikrishna715@gmail.com for licensing information.
    
    Args:
        num_qubits: Number of qubits in the quantum system
        backend: Quantum simulation backend ('default.qubit', 'lightning.qubit')
        shots: Number of measurement shots for probabilistic outcomes
        noise_model: Optional noise model for realistic simulation
    """
    
    def __init__(
        self,
        num_qubits: int = 8,
        backend: str = "default.qubit",
        shots: Optional[int] = None,
        noise_model: Optional[Dict] = None,
    ):
        # Validate license before allowing class instantiation
        validate_class_license(["quantum_simulator"])
        
        self.num_qubits = num_qubits
        self.backend = backend
        self.shots = shots
        self.noise_model = noise_model
        
        # Initialize PennyLane device
        self.device = qml.device(backend, wires=num_qubits, shots=shots)
        
        # Track quantum circuit history
        self.circuit_history = []
        self.measurement_history = []
        
        # Noise parameters
        self.noise_strength = noise_model.get('strength', 0.01) if noise_model else 0.0
        self.decoherence_rate = noise_model.get('decoherence', 0.001) if noise_model else 0.0
    
    def create_quantum_circuit(self, operations: List[Dict[str, Any]]) -> Any:
        """
        Create a quantum circuit from operation specifications.
        
        Args:
            operations: List of quantum operations
                Each operation is a dict with 'gate', 'wires', and optional 'params'
                
        Returns:
            PennyLane QNode representing the quantum circuit if available, 
            otherwise classical simulation
        """
        if not PENNYLANE_AVAILABLE:
            # Return simplified classical circuit simulation
            return self._create_classical_circuit(operations)
            
        def circuit():
            for op in operations:
                gate_name = op['gate']
                wires = op['wires']
                params = op.get('params', [])
                
                # Apply quantum gate
                if gate_name == 'X':
                    qml.PauliX(wires=wires)
                elif gate_name == 'Y':
                    qml.PauliY(wires=wires)
                elif gate_name == 'Z':
                    qml.PauliZ(wires=wires)
                elif gate_name == 'H':
                    qml.Hadamard(wires=wires)
                elif gate_name == 'RX':
                    qml.RX(params[0], wires=wires)
                elif gate_name == 'RY':
                    qml.RY(params[0], wires=wires)
                elif gate_name == 'RZ':
                    qml.RZ(params[0], wires=wires)
                elif gate_name == 'CNOT':
                    qml.CNOT(wires=wires)
                elif gate_name == 'CZ':
                    qml.CZ(wires=wires)
                elif gate_name == 'SWAP':
                    qml.SWAP(wires=wires)
                elif gate_name == 'Toffoli':
                    qml.Toffoli(wires=wires)
                else:
                    raise ValueError(f"Unsupported gate: {gate_name}")
                
                # Apply noise if enabled
                if self.noise_strength > 0:
                    self._apply_noise(wires if isinstance(wires, list) else [wires])
            
            # Return state vector or expectation values
            return qml.state()
        
        if PENNYLANE_AVAILABLE:
            qnode = qml.QNode(circuit, self.device)
        else:
            qnode = self._create_classical_circuit(operations)
        self.circuit_history.append(operations)
        return qnode
    
    def _apply_noise(self, wires: List[int]) -> None:
        """Apply noise operations to specified wires."""
        for wire in wires:
            if wire < self.num_qubits:
                # Depolarizing noise
                if np.random.random() < self.noise_strength:
                    noise_gate = np.random.choice(['X', 'Y', 'Z'])
                    if noise_gate == 'X':
                        qml.PauliX(wires=wire)
                    elif noise_gate == 'Y':
                        qml.PauliY(wires=wire)
                    elif noise_gate == 'Z':
                        qml.PauliZ(wires=wire)
                
                # Phase damping (simplified decoherence)
                if np.random.random() < self.decoherence_rate:
                    qml.RZ(np.random.normal(0, 0.1), wires=wire)
    
    def encode_classical_data(
        self, 
        data: Union[torch.Tensor, np.ndarray], 
        encoding_type: str = "amplitude"
    ) -> List[Dict[str, Any]]:
        """
        Encode classical data into quantum circuit operations.
        
        Args:
            data: Classical data to encode (shape: [..., features])
            encoding_type: Type of encoding ('amplitude', 'angle', 'basis')
            
        Returns:
            List of quantum operations for data encoding
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        # Flatten and normalize data
        flat_data = data.flatten()
        
        operations = []
        
        if encoding_type == "amplitude":
            # Amplitude encoding: encode data in quantum state amplitudes
            # Normalize data to unit vector
            norm = np.linalg.norm(flat_data)
            if norm > 0:
                normalized_data = flat_data / norm
            else:
                normalized_data = flat_data
            
            # Create superposition state with data-dependent amplitudes
            for i, amplitude in enumerate(normalized_data[:2**self.num_qubits]):
                if abs(amplitude) > 1e-6:  # Avoid near-zero amplitudes
                    # Convert amplitude to rotation angle
                    angle = 2 * np.arcsin(min(abs(amplitude), 1.0))
                    
                    # Apply rotation to create desired amplitude
                    qubit_idx = i % self.num_qubits
                    operations.append({
                        'gate': 'RY',
                        'wires': qubit_idx,
                        'params': [angle]
                    })
        
        elif encoding_type == "angle":
            # Angle encoding: encode data in rotation angles
            for i, value in enumerate(flat_data[:self.num_qubits]):
                # Scale value to appropriate range for rotation
                angle = value * np.pi / 2  # Scale to [-π/2, π/2]
                operations.append({
                    'gate': 'RY',
                    'wires': i,
                    'params': [angle]
                })
        
        elif encoding_type == "basis":
            # Basis encoding: encode data in computational basis
            # Convert data to binary representation
            for i, value in enumerate(flat_data[:self.num_qubits]):
                if value > 0.5:  # Threshold for |1⟩ state
                    operations.append({
                        'gate': 'X',
                        'wires': i,
                        'params': []
                    })
        
        return operations
    
    def create_entanglement_circuit(
        self, 
        entanglement_pattern: str = "linear",
        entanglement_strength: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Create quantum circuit for generating entanglement.
        
        Args:
            entanglement_pattern: Pattern of entanglement ('linear', 'circular', 'all-to-all')
            entanglement_strength: Strength of entanglement (0-1)
            
        Returns:
            List of quantum operations for entanglement generation
        """
        operations = []
        
        # Create superposition states
        for i in range(self.num_qubits):
            operations.append({
                'gate': 'H',
                'wires': i,
                'params': []
            })
        
        # Apply entangling gates based on pattern
        if entanglement_pattern == "linear":
            # Linear chain of CNOT gates
            for i in range(self.num_qubits - 1):
                operations.append({
                    'gate': 'CNOT',
                    'wires': [i, i + 1],
                    'params': []
                })
                
                # Add rotation for partial entanglement
                if entanglement_strength < 1.0:
                    angle = entanglement_strength * np.pi / 2
                    operations.append({
                        'gate': 'RY',
                        'wires': i + 1,
                        'params': [angle]
                    })
        
        elif entanglement_pattern == "circular":
            # Circular entanglement pattern
            for i in range(self.num_qubits):
                next_qubit = (i + 1) % self.num_qubits
                operations.append({
                    'gate': 'CNOT',
                    'wires': [i, next_qubit],
                    'params': []
                })
        
        elif entanglement_pattern == "all-to-all":
            # All-to-all entanglement (exponentially complex)
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    operations.append({
                        'gate': 'CNOT',
                        'wires': [i, j],
                        'params': []
                    })
                    
                    # Add controlled rotation for complex entanglement
                    angle = entanglement_strength * np.pi / (2 * self.num_qubits)
                    operations.append({
                        'gate': 'RZ',
                        'wires': j,
                        'params': [angle]
                    })
        
        return operations
    
    def measure_expectation_values(
        self, 
        circuit_ops: List[Dict[str, Any]], 
        observables: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Measure expectation values of quantum observables.
        
        Args:
            circuit_ops: Quantum circuit operations
            observables: List of observables to measure ('X', 'Y', 'Z', 'H')
            
        Returns:
            Dictionary mapping observable names to expectation values
        """
        if observables is None:
            observables = ['Z'] * self.num_qubits
        
        expectation_values = {}
        
        for i, obs in enumerate(observables):
            if i >= self.num_qubits:
                break
            
            def circuit():
                # Apply circuit operations
                for op in circuit_ops:
                    gate_name = op['gate']
                    wires = op['wires']
                    params = op.get('params', [])
                    
                    if gate_name == 'X':
                        qml.PauliX(wires=wires)
                    elif gate_name == 'Y':
                        qml.PauliY(wires=wires)
                    elif gate_name == 'Z':
                        qml.PauliZ(wires=wires)
                    elif gate_name == 'H':
                        qml.Hadamard(wires=wires)
                    elif gate_name == 'RX':
                        qml.RX(params[0], wires=wires)
                    elif gate_name == 'RY':
                        qml.RY(params[0], wires=wires)
                    elif gate_name == 'RZ':
                        qml.RZ(params[0], wires=wires)
                    elif gate_name == 'CNOT':
                        qml.CNOT(wires=wires)
                
                # Return expectation value
                if obs == 'X':
                    return qml.expval(qml.PauliX(i))
                elif obs == 'Y':
                    return qml.expval(qml.PauliY(i))
                elif obs == 'Z':
                    return qml.expval(qml.PauliZ(i))
                elif obs == 'H':
                    return qml.expval(qml.Hadamard(i))
            
            qnode = qml.QNode(circuit, self.device)
            expectation_values[f"{obs}_{i}"] = float(qnode())
        
        self.measurement_history.append(expectation_values)
        return expectation_values
    
    def compute_entanglement_measures(
        self, 
        state_vector: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute various entanglement measures from quantum state.
        
        Args:
            state_vector: Quantum state vector (if None, uses last circuit result)
            
        Returns:
            Dictionary with entanglement measures
        """
        if state_vector is None:
            # Get state from last circuit
            if not self.circuit_history:
                raise ValueError("No circuit history available")
            
            last_ops = self.circuit_history[-1]
            circuit = self.create_quantum_circuit(last_ops)
            state_vector = circuit()
        
        # Convert to numpy if necessary
        if hasattr(state_vector, 'numpy'):
            state_vector = state_vector.numpy()
        
        measures = {}
        
        # Von Neumann entropy (for subsystems)
        if self.num_qubits >= 2:
            # Compute reduced density matrix for first qubit
            num_states = 2 ** self.num_qubits
            state_matrix = np.outer(state_vector, np.conj(state_vector))
            
            # Trace out all qubits except the first
            reduced_dm = np.zeros((2, 2), dtype=complex)
            for i in range(0, num_states, 2):
                for j in range(0, num_states, 2):
                    reduced_dm[0, 0] += state_matrix[i, j]
                    reduced_dm[0, 1] += state_matrix[i, j + 1]
                    reduced_dm[1, 0] += state_matrix[i + 1, j]
                    reduced_dm[1, 1] += state_matrix[i + 1, j + 1]
            
            # Compute eigenvalues for entropy
            eigenvals = np.linalg.eigvals(reduced_dm)
            eigenvals = eigenvals[eigenvals > 1e-12]  # Filter out zero eigenvalues
            entropy = -np.sum(eigenvals * np.log2(eigenvals))
            measures["von_neumann_entropy"] = float(entropy)
        
        # Concurrence (for two-qubit systems)
        if self.num_qubits == 2:
            # Compute concurrence for two-qubit state
            pauli_y = np.array([[0, -1j], [1j, 0]])
            spin_flipped = np.kron(pauli_y, pauli_y).dot(np.conj(state_vector))
            concurrence = abs(np.vdot(state_vector, spin_flipped))
            measures["concurrence"] = float(concurrence)
        
        # Participation ratio
        participation_ratio = 1.0 / np.sum(np.abs(state_vector) ** 4)
        measures["participation_ratio"] = float(participation_ratio)
        
        # Linear entropy
        state_matrix = np.outer(state_vector, np.conj(state_vector))
        linear_entropy = 1.0 - np.trace(state_matrix @ state_matrix)
        measures["linear_entropy"] = float(linear_entropy.real)
        
        return measures
    
    def simulate_decoherence(
        self, 
        initial_state: np.ndarray, 
        time_steps: int = 100,
        decoherence_model: str = "amplitude_damping"
    ) -> List[np.ndarray]:
        """
        Simulate quantum decoherence over time.
        
        Args:
            initial_state: Initial quantum state vector
            time_steps: Number of time steps for simulation
            decoherence_model: Type of decoherence ('amplitude_damping', 'phase_damping', 'depolarizing')
            
        Returns:
            List of state vectors showing decoherence evolution
        """
        states = [initial_state.copy()]
        current_state = initial_state.copy()
        
        for t in range(time_steps):
            if decoherence_model == "amplitude_damping":
                # Amplitude damping: |1⟩ → |0⟩ transitions
                damping_rate = self.decoherence_rate
                for i in range(self.num_qubits):
                    # Apply amplitude damping to each qubit
                    qubit_state = self._extract_qubit_state(current_state, i)
                    damped_state = self._apply_amplitude_damping(qubit_state, damping_rate)
                    current_state = self._update_qubit_state(current_state, damped_state, i)
            
            elif decoherence_model == "phase_damping":
                # Phase damping: loss of coherence between |0⟩ and |1⟩
                for i in range(self.num_qubits):
                    phase_noise = np.random.normal(0, self.decoherence_rate)
                    current_state = self._apply_phase_noise(current_state, i, phase_noise)
            
            elif decoherence_model == "depolarizing":
                # Depolarizing noise: random Pauli operations
                for i in range(self.num_qubits):
                    if np.random.random() < self.decoherence_rate:
                        pauli_op = np.random.choice(['X', 'Y', 'Z'])
                        current_state = self._apply_pauli_to_state(current_state, i, pauli_op)
            
            # Normalize state
            current_state = current_state / np.linalg.norm(current_state)
            states.append(current_state.copy())
        
        return states
    
    def _extract_qubit_state(self, full_state: np.ndarray, qubit_idx: int) -> np.ndarray:
        """Extract single qubit state from multi-qubit state."""
        # Simplified extraction - for full implementation, would need proper tracing
        return np.array([full_state[0], full_state[1]]) / np.linalg.norm([full_state[0], full_state[1]])
    
    def _apply_amplitude_damping(self, qubit_state: np.ndarray, gamma: float) -> np.ndarray:
        """Apply amplitude damping to single qubit state."""
        # Kraus operators for amplitude damping
        E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        E1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        
        # Apply Kraus operators (simplified)
        new_state = E0.dot(qubit_state)
        return new_state / np.linalg.norm(new_state)
    
    def _update_qubit_state(
        self, full_state: np.ndarray, qubit_state: np.ndarray, qubit_idx: int
    ) -> np.ndarray:
        """Update single qubit in multi-qubit state."""
        # Simplified update - for full implementation, would need tensor product operations
        updated_state = full_state.copy()
        updated_state[:2] = qubit_state * np.linalg.norm(updated_state[:2])
        return updated_state / np.linalg.norm(updated_state)
    
    def _apply_phase_noise(
        self, state: np.ndarray, qubit_idx: int, phase: float
    ) -> np.ndarray:
        """Apply phase noise to specific qubit."""
        # Simplified phase noise application
        noisy_state = state.copy()
        for i in range(len(state)):
            if (i >> qubit_idx) & 1:  # If qubit is in |1⟩ state
                noisy_state[i] *= np.exp(1j * phase)
        return noisy_state
    
    def _apply_pauli_to_state(
        self, state: np.ndarray, qubit_idx: int, pauli_op: str
    ) -> np.ndarray:
        """Apply Pauli operation to specific qubit in multi-qubit state."""
        # Simplified Pauli application
        if pauli_op == 'X':
            # Bit flip
            new_state = state.copy()
            for i in range(len(state)):
                flipped_i = i ^ (1 << qubit_idx)
                new_state[i] = state[flipped_i]
            return new_state
        elif pauli_op == 'Z':
            # Phase flip
            new_state = state.copy()
            for i in range(len(state)):
                if (i >> qubit_idx) & 1:
                    new_state[i] *= -1
            return new_state
        elif pauli_op == 'Y':
            # Y = iXZ
            state_x = self._apply_pauli_to_state(state, qubit_idx, 'X')
            state_y = self._apply_pauli_to_state(state_x, qubit_idx, 'Z')
            return 1j * state_y
        
        return state
    
    def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the quantum simulation."""
        return {
            "num_qubits": self.num_qubits,
            "backend": self.backend,
            "circuits_executed": len(self.circuit_history),
            "measurements_performed": len(self.measurement_history),
            "noise_strength": self.noise_strength,
            "decoherence_rate": self.decoherence_rate,
            "device_capabilities": str(self.device),
        }
