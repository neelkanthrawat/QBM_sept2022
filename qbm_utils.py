##########################################################################################
                                        ## imports ##
###########################################################################################



# from tkinter.tix import Tree
import numpy as np
from numpy import pi
import math
import seaborn as sns
from IPython.display import Image
import matplotlib.pyplot as plt
from typing import Mapping, Union, Iterable, Optional

from qiskit import *
from qiskit.circuit.library import *
from qiskit.algorithms import *
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit #, InstructionSet
from qiskit import quantum_info, IBMQ, Aer
from qiskit.quantum_info import partial_trace, Statevector, state_fidelity, Pauli
from qiskit.utils import QuantumInstance
from qiskit.extensions import HamiltonianGate, RZZGate, RZGate, RXGate
from qiskit.circuit.quantumregister import Qubit
from qiskit.visualization import plot_histogram, plot_state_qsphere, plot_bloch_multivector, plot_bloch_vector

qsm = Aer.get_backend('qasm_simulator')
stv = Aer.get_backend('statevector_simulator')
aer = Aer.get_backend('aer_simulator')


##################################################################################################
                                    ## helper functions ##
##################################################################################################


def measure_and_plot(qc: QuantumCircuit, shots:int= 1024, show_counts:bool= False, return_counts:bool= False, measure_cntrls: bool = False, decimal_count_keys:bool = True , cntrl_specifier:Union[int, list, str] = 'all'):
    """ Measure and plot the state of the data registers, optionally measure the control ancillas, without modifying the original circuit.
        
        ARGS:
        ----
            qc : 'QuantumCircuit' 
                 the circuit to be measured

            shots: 'int' 
                    no. of shots for the measurement

            show_counts : 'bool' 
                           print the counts dictionary

            measure_cntrls : 'bool' 
                             indicates whether to measure the control ancilla registers.
            
            return_counts : 'bool'
                            returns the counts obtained if True, else retruns the histogram plot

            measure_cntrls: 'bool'
                             indicates whether to measure the controll ancill qubits
                
            decimal_count_keys: 'bool'
                                if 'True' converts the binary state of the controll ancilllas to integer represntation

            cntrl_specifier : 'int' 
                                inidicates whihch of the control registers to meausure, 
                                for eg. cntrl_specifier= 2 refers to the first control ancilla cntrl_2
                                cntrl_specifier= 'all' refers to all the ancillas                                                           
        RETURNS:
        -------
            plots histogram over the computational basis states

     """
    qc_m = qc.copy()
    creg = ClassicalRegister( len(qc_m.qregs[0]) )
    qc_m.add_register(creg)
    qc_m.measure(qc_m.qregs[0], creg)

    if measure_cntrls== True:

        if isinstance(cntrl_specifier, int):
            print('int')##cflag
            if cntrl_specifier > len(qc_m.qregs) or cntrl_specifier < 1: raise ValueError(" 'cntrl_specifier' should be less than no. of control registers and greater than 0")

            creg_cntrl = ClassicalRegister(len(qc_m.qregs[cntrl_specifier-1]))
            qc_m.add_register(creg_cntrl)
            qc_m.measure(qc_m.qregs[cntrl_specifier-1], creg_cntrl )

        elif isinstance(cntrl_specifier, list ):
            print('list')##cflag
            for ancilla in cntrl_specifier:

                if ancilla > len(qc_m.qregs) or ancilla < 1: raise ValueError(" 'ancilla' should be less than no. of control registers and greater than 0")

                creg_cntrl = ClassicalRegister(len(qc_m.qregs[ancilla-1]))
                qc_m.add_register(creg_cntrl)
                qc_m.measure(qc_m.qregs[ancilla-1], creg_cntrl )

        elif isinstance(cntrl_specifier, str) and cntrl_specifier== "all":
            print('str')##cflag
            for reg in qc_m.qregs[1:] :
                creg = ClassicalRegister(len(reg))
                qc_m.add_register(creg)
                qc_m.measure(reg, creg)
                
    # plt.figure()
    counts = execute(qc_m, qsm, shots= shots).result().get_counts()
   
    if decimal_count_keys:
        counts_m = {}
        for key in counts.keys():
            split_key = key.split(' ')
            if measure_cntrls:
                key_m = 'key: '
                for string in split_key[:-1]:
                    key_m+= str(int(string, 2)) + ' '
                key_m += '-> '
            else: key_m = ''
            key_m += split_key[-1][::-1] ##to-check
            counts_m[key_m] = counts[key]
        counts = counts_m

    if show_counts== True: print(counts)
    
    if return_counts: return counts
    else: return plot_histogram(counts)
 


def construct_h1(a:float, b:float, time:float= 1.00 ):
    """ Function to construct a circuit implementing single qubit unitaries
        i.e;  H1_i = a_i * X_i + b_i * Z_i

        ARGS:
        ----
            a: [float] 
                Coefficient of 'X' operator

            b: [float] 
                Coefficient of 'Z' operator
            
            time: Optional[float]
                    Time of evolution while implemnting using 'HamiltonianGate' class 
        
        RETURNS:
        -------
            qiskit.HamiltonianGate object 

    """

    return HamiltonianGate( a * Pauli('X').to_matrix() + b * Pauli('Z').to_matrix(), time )


def construct_h2(J:float, time:float = 1.00 ):
    """ Function to construct circuit implmenting RZZ rotations 

    """
    pass

def append_evolution(qc:QuantumCircuit, h:np.array , J:np.array, gamma:float, alpha:float, time:float, is_terminal_step= False):

    for qubit in range(len(qc.qubits)):
        qc.append(HamiltonianGate( gamma * Pauli('X').to_matrix() + (1 - gamma) * alpha * h[qubit] * Pauli('Z').to_matrix(), time, label= 'h_'+str(qubit) ), [ qc.qubits[qubit]] )
    
    if not is_terminal_step:    
        for qubit in range( 0, len(qc.qubits), 2):
            qc.append( RZZGate(J[qubit][qubit+1], label= 'J_'+str(qubit)+str(qubit+1)), [qc.qubits[qubit], qc.qubits[qubit+1]]  )
        for qubit in range( 1, len(qc.qubits)-1 , 2):
            qc.append( RZZGate(J[qubit][qubit+1], label= 'J_'+str(qubit)+str(qubit+1)), [qc.qubits[qubit], qc.qubits[qubit+1]]  )
    
    qc.barrier()
    return qc

class IsingEnergyFunction():
    """ A class to build the Ising Hamiltonian from data
    
    """
    def __init__(self, J: np.array, h: np.array) -> None:
        self.J = J
        self.h = h
        self.num_spins = len(h)
    
    def get_energy(self, state:Union[str, np.array] )-> float:

        if isinstance(state, str):
            state = np.array( [int(list(s)[i]) for i in range(len(s))])
            energy =  np.dot(state.transpose(), self.J.dot(state)) + np.dot(self.h.transpose(), state )
            return energy
        else:
            return np.dot(state.transpose(), self.J.dot(state)) + np.dot(self.h.transpose(), state )
    
    def get_boltzmann_prob(self, state:Union[str, np.array], beta:float= 1.0) -> float:

        return np.exp( -1 * beta * self.get_energy(state) )