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
from collections import Counter

from qiskit import *
from qiskit.circuit.library import *
from qiskit.algorithms import *
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit #, InstructionSet
from qiskit import quantum_info, IBMQ, Aer
from qiskit.quantum_info import partial_trace, Statevector, state_fidelity, Pauli
from qiskit.utils import QuantumInstance
from qiskit.extensions import HamiltonianGate
from qiskit.circuit.library import RZZGate, RZGate, RXGate
from qiskit.circuit.quantumregister import Qubit
from qiskit.visualization import plot_histogram, plot_state_qsphere, plot_bloch_multivector, plot_bloch_vector

qsm = Aer.get_backend('qasm_simulator')
stv = Aer.get_backend('statevector_simulator')
aer = Aer.get_backend('aer_simulator')


##################################################################################################
                                    ## decorator functions ##
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
 
################################################################################################
                                    ##  helper functions  ##
################################################################################################

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

    for qubit in range(len(qc.qubits)):# H1
        qc.append(HamiltonianGate( gamma * Pauli('X').to_matrix() + (1 - gamma) * alpha * h[qubit] * Pauli('Z').to_matrix(), time, label= 'h_'+str(qubit) ), [ qc.qubits[qubit]] )
    
    if not is_terminal_step:   # H2
        for qubit in range( 0, len(qc.qubits), 2):
            qc.append( RZZGate(J[qubit][qubit+1], label= 'J_'+str(qubit)+str(qubit+1)), [qc.qubits[qubit], qc.qubits[qubit+1]]  )
        for qubit in range( 1, len(qc.qubits)-1 , 2):
            qc.append( RZZGate(J[qubit][qubit+1], label= 'J_'+str(qubit)+str(qubit+1)), [qc.qubits[qubit], qc.qubits[qubit+1]]  )
    
    qc.barrier()
    return qc


def initialise_qc(n_spins:int, bitstring:str):
  '''
  Initialises a quantum circuit with n_spins number of qubits in a state defined by "bitstring"
  (Caution: Qiskit's indexing convention for qubits (order of tensor product) is different from the conventional textbook one!)  
  '''

  spins = QuantumRegister(n_spins, name= 'spin')
  creg_final = ClassicalRegister(n_spins, name= 'creg_f')
  qc_in = QuantumCircuit(spins, creg_final)

  len_str_in=len(bitstring)
  assert(len_str_in == len(qc_in.qubits)), "len(bitstring) should be equal to number_of_qubits/spins"

  #print("qc_in.qubits: ", qc_in.qubits)
  where_x_gate=[qc_in.qubits[len_str_in-1-i] for i in range(0,len(bitstring)) if bitstring[i]=='1']
  if len(where_x_gate)!=0:
    qc_in.x(where_x_gate)
  return qc_in, spins, creg_final


class IsingEnergyFunction():
    """ A class to build the Ising Hamiltonian from data
    
    """
    def __init__(self, J: np.array, h: np.array) -> None:
        self.J = J
        self.h = h
        self.num_spins = len(h)
    
    def get_energy(self, state:Union[str, np.array] )-> float:

        if isinstance(state, str):
            state = np.array( [int(list(state)[i]) for i in range(len(state))])
            energy =  np.dot(state.transpose(), self.J.dot(state)) + np.dot(self.h.transpose(), state )
            return energy
        else:
            return np.dot(state.transpose(), self.J.dot(state)) + np.dot(self.h.transpose(), state )
    
    def get_boltzmann_prob(self, state:Union[str, np.array], beta:float= 1.0) -> float:

        return np.exp( -1 * beta * self.get_energy(state) )

################################################################################################
                                ##  Classical MCMC routines ##
################################################################################################

def classical_transition(num_spins:int)->str:
    '''   
    Returns s' , obtained via uniform transition rule!
    '''
    num_elems=2**(num_spins)
    next_state=np.random.randint(0,num_elems)# since upper limit is exclusive and lower limit is inclusive
    bin_next_state=f'{next_state:0{num_spins}b}'
    return bin_next_state

def classical_loop_accepting_state(s_init:str, s_prime:str,energy_s:float,energy_sprime:float,temp=1)->str:
    '''  
    Accepts the state sprime with probability A ( i.e. min(1,exp(-(E(s')-E(s))/ temp) ) 
    and s_init with probability 1-A
    '''
    delta_energy=energy_sprime-energy_s # E(s')-E(s)
    exp_factor=np.exp(-delta_energy/temp)
    acceptance=min(1,exp_factor)# for both QC case as well as uniform random strategy, the transition matrix Pij is symmetric!
    coin_flip=np.random.choice([True, False], p=[acceptance, 1-acceptance])
    new_state=s_init
    if coin_flip:
        new_state=s_prime
    
    return new_state

def classical_mcmc(N_hops:int, num_spins:int, num_elems:int, model, return_last_n_states=500):
    ''' 
    Args: 
    Nhops: Number of time you want to run mcmc
    num_spins: number of spins
    num_elems: 2**(num_spins)
    model

    Returns:
    Last 'return_last_n_states' elements of states so collected (default value=500). one can then deduce the distribution from it! 
    '''
    states=[]
    current_state=f'{np.random.randint(0,num_elems):0{num_spins}b}'# bin_next_state=f'{next_state:0{num_spins}b}'
    print("starting with: ", current_state) 
    for i in range(0, N_hops):
        states.append(current_state)
        # get sprime
        s_prime=classical_transition(num_spins)
        # accept/reject s_prime 
        energy_s=model.get_energy(current_state)
        energy_sprime=model.get_energy(s_prime)
        next_state= classical_loop_accepting_state(current_state, s_prime, energy_s, energy_sprime,temp=1)
        current_state= next_state
    
    return Counter(states[-return_last_n_states:])# returns dictionary of occurences for last 500 states

################################################################################################
                                ##  Qunatum MCMC routines ##
################################################################################################

def run_qc_quantum_step(qc_initialised_to_s:QuantumCircuit, h, J, alpha,n_spins:int, num_trotter_steps=10, time=0.8)->str:

    '''     
    Takes in a qc initialized to some state "s". After performing unitary evolution U=exp(-iHt)
    , circuit is measured once and returns the bitstring ,s', corresponding to the measured state .

    Args:
    qc_initialised_to_s
    alpha:
    num_trotter_steps: (default 10)
    time: For how long you want to evolve.
    '''
    for _ in range(num_trotter_steps):
        append_evolution(qc_initialised_to_s, h, J ,gamma=np.random.random(), alpha=alpha, time=time)
    append_evolution(qc_initialised_to_s, h, J , gamma=0.1, alpha=alpha, time=time, is_terminal_step=True)
    
    # draw the ckt
    #print(qc_initialised_to_s.draw())

    # run the circuit
    #creg_final=ClassicalRegister(n_spins, name= 'creg_f')
    num_shots=1
    qc_initialised_to_s.measure(qc_initialised_to_s.qregs[0], qc_initialised_to_s.cregs[0])
    state_obtained_dict=execute(qc_initialised_to_s, shots= num_shots, backend= qsm).result().get_counts()
    state_obtained=list(state_obtained_dict.keys())[0]#since there is only one element
    return state_obtained


def quantum_enhanced_mcmc(N_hops:int, num_spins:int, num_elems:int,model, h,J, alpha, num_trotter_steps=10, return_last_n_states=500):
    ''' 
    Args: 
    Nhops: Number of time you want to run mcmc
    num_spins: number of spins
    num_elems: 2**(num_spins)

    Returns:
    Last 'return_last_n_states' elements of states so collected (default value=500). one can then deduce the distribution from it! 
    '''
    states=[]
    current_state=f'{np.random.randint(0,num_elems):0{num_spins}b}'# bin_next_state=f'{next_state:0{num_spins}b}'
    print("starting with: ", current_state) 
    # initialise quantum circuit to current_state
    qc_s=initialise_qc(n_spins=num_spins, bitstring=current_state)
    for i in range(0, N_hops):
        states.append(current_state)
        # get sprime
        s_prime=run_qc_quantum_step(qc_initialised_to_s=qc_s, h=h, J=J, alpha=alpha, n_spins=num_spins, num_trotter_steps=num_trotter_steps, time=0.8)
        # accept/reject s_prime 
        energy_s=model.get_energy(current_state)
        energy_sprime=model.get_energy(s_prime)
        next_state= classical_loop_accepting_state(current_state, s_prime, energy_s, energy_sprime,temp=1)
        if current_state!=next_state:
          current_state= next_state
        qc_s=initialise_qc(n_spins=num_spins, bitstring=current_state)
    
    return Counter(states[-return_last_n_states:])# returns dictionary of occurences for last 500 states