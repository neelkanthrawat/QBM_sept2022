#########################################################################################
## imports ##
###########################################################################################


import itertools
import math
from collections import Counter
from typing import Iterable, Mapping, Optional, Union
import numba 
from numba import jit, types, typed 
from numba.experimental import jitclass

import matplotlib.pyplot as plt

# from tkinter.tix import Tree
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image
from numpy import pi
from qiskit import *
from qiskit import (  # , InstructionSet
    IBMQ,
    Aer,
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    quantum_info,
)
from qiskit.algorithms import *
from qiskit.circuit.library import *
from qiskit.circuit.library import RXGate, RZGate, RZZGate
from qiskit.circuit.quantumregister import Qubit
from qiskit.extensions import HamiltonianGate, UnitaryGate
from qiskit.quantum_info import Pauli, Statevector, partial_trace, state_fidelity
from qiskit.utils import QuantumInstance
from qiskit.visualization import (
    plot_bloch_multivector,
    plot_bloch_vector,
    plot_histogram,
    plot_state_qsphere,
)
from tqdm import tqdm

# import function


qsm = Aer.get_backend("qasm_simulator")
stv = Aer.get_backend("statevector_simulator")
aer = Aer.get_backend("aer_simulator")


##################################################################################################
## decorator functions ##
##################################################################################################


def measure_and_plot(
    qc: QuantumCircuit,
    shots: int = 1024,
    show_counts: bool = False,
    return_counts: bool = False,
    measure_cntrls: bool = False,
    decimal_count_keys: bool = True,
    cntrl_specifier: Union[int, list, str] = "all",
):
    """Measure and plot the state of the data registers, optionally measure the control ancillas, without modifying the original circuit.

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
    creg = ClassicalRegister(len(qc_m.qregs[0]))
    qc_m.add_register(creg)
    qc_m.measure(qc_m.qregs[0], creg)

    if measure_cntrls == True:

        if isinstance(cntrl_specifier, int):
            print("int")  ##cflag
            if cntrl_specifier > len(qc_m.qregs) or cntrl_specifier < 1:
                raise ValueError(
                    " 'cntrl_specifier' should be less than no. of control registers and greater than 0"
                )

            creg_cntrl = ClassicalRegister(len(qc_m.qregs[cntrl_specifier - 1]))
            qc_m.add_register(creg_cntrl)
            qc_m.measure(qc_m.qregs[cntrl_specifier - 1], creg_cntrl)

        elif isinstance(cntrl_specifier, list):
            print("list")  ##cflag
            for ancilla in cntrl_specifier:

                if ancilla > len(qc_m.qregs) or ancilla < 1:
                    raise ValueError(
                        " 'ancilla' should be less than no. of control registers and greater than 0"
                    )

                creg_cntrl = ClassicalRegister(len(qc_m.qregs[ancilla - 1]))
                qc_m.add_register(creg_cntrl)
                qc_m.measure(qc_m.qregs[ancilla - 1], creg_cntrl)

        elif isinstance(cntrl_specifier, str) and cntrl_specifier == "all":
            print("str")  ##cflag
            for reg in qc_m.qregs[1:]:
                creg = ClassicalRegister(len(reg))
                qc_m.add_register(creg)
                qc_m.measure(reg, creg)

    # plt.figure()
    counts = execute(qc_m, qsm, shots=shots).result().get_counts()

    if decimal_count_keys:
        counts_m = {}
        for key in counts.keys():
            split_key = key.split(" ")
            if measure_cntrls:
                key_m = "key: "
                for string in split_key[:-1]:
                    key_m += str(int(string, 2)) + " "
                key_m += "-> "
            else:
                key_m = ""
            key_m += split_key[-1][::-1]  ##to-check
            counts_m[key_m] = counts[key]
        counts = counts_m

    if show_counts == True:
        print(counts)

    if return_counts:
        return counts
    else:
        return plot_histogram(counts)


################################################################################################
##  helper functions  ##
################################################################################################
# def initialise_qc(n_spins:int, bitstring:str):
#   '''
#   Initialises a quantum circuit with n_spins number of qubits in a state defined by "bitstring"
#   (Caution: Qiskit's indexing convention for qubits (order of tensor product) is different from the conventional textbook one!)
#   '''

#   spins = QuantumRegister(n_spins, name= 'spin')
#   creg_final = ClassicalRegister(n_spins, name= 'creg_f')
#   qc_in = QuantumCircuit(spins, creg_final)

#   len_str_in=len(bitstring)
#   assert(len_str_in == len(qc_in.qubits)), "len(bitstring) should be equal to number_of_qubits/spins"

#   #print("qc_in.qubits: ", qc_in.qubits)
#   where_x_gate=[qc_in.qubits[len_str_in-1-i] for i in range(0,len(bitstring)) if bitstring[i]=='1']
#   if len(where_x_gate)!=0:
#     qc_in.x(where_x_gate)
#   return qc_in


# @jitclass
class IsingEnergyFunction:
    """A class to build the Ising Hamiltonian from data"""

    def __init__(self, J: np.array, h: np.array, beta: float = 1.0) -> None:
        self.J = J
        self.h = h
        self.beta = beta
        self.num_spins = len(h)

    def get_J(self):
        return self.J

    def get_h(self):
        return self.h

    @jit(nopython= True)
    def get_energy(self, state: Union[str, np.array]) -> float:
        """'state' should be a bipolar state if it is an array"""

        if isinstance(state, str):
            state = np.array([-1 if elem == "0" else 1 for elem in state])
            # state = np.array( [int(list(state)[i]) for i in range(len(state))])
            energy = 0.5 * np.dot(state.transpose(), self.J.dot(state)) + np.dot(
                self.h.transpose(), state
            )
            return energy
        else:
            return 0.5 * np.dot(state.transpose(), self.J.dot(state)) + np.dot(
                self.h.transpose(), state
            )
    @jit(nopython= True)
    def get_partition_sum(self, beta: float = 1.0):  ## is computationally expensive

        all_configs = np.array(list(itertools.product([1, 0], repeat=self.num_spins)))
        return sum([self.get_boltzmann_prob(configbeta=beta) for config in all_configs])

    @jit(nopython= True)
    def get_boltzmann_prob(
        self, state: Union[str, np.array], beta: float = 1.0, normalised=False
    ) -> float:

        if normalised:
            return np.exp(-1 * beta * self.get_energy(state)) / self.get_partition_sum(
                beta
            )

        else:
            return np.exp(-1 * beta * self.get_energy(state))

    def get_observable_expectation(self, observable, beta: float = 1.0) -> float:
        """Return expectation value of a classical observables

        ARGS :
        ----
        observable: Must be a function of the spin configuration which takes an 'np.array' of binary elements as input argument and returns a 'float'
        beta: inverse temperature

        """
        all_configs = np.array(list(itertools.product([1, 0], repeat=self.num_spins)))
        partition_sum = sum([self.get_boltzmann_prob(config) for config in all_configs])

        return sum(
            [
                self.get_boltzmann_prob(config)
                * observable(config)
                * (1 / partition_sum)
                for config in all_configs
            ]
        )

    def get_entropy(self, beta: float = 1.00) -> float:

        return np.log(
            self.get_partition_sum(beta)
        ) - beta * self.get_observable_expectation(self.get_energy)

    def get_kldiv(self, prob_dict: dict):

        pass


################################################################################################
##  Classical MCMC routines ##
################################################################################################


def classical_transition(num_spins: int) -> str:
    """
    Returns s' , obtained via uniform transition rule!
    """
    num_elems = 2 ** (num_spins)
    next_state = np.random.randint(
        0, num_elems
    )  # since upper limit is exclusive and lower limit is inclusive
    bin_next_state = f"{next_state:0{num_spins}b}"
    return bin_next_state


def classical_loop_accepting_state(
    s_init: str, s_prime: str, energy_s: float, energy_sprime: float, temp=1
) -> str:
    """
    Accepts the state sprime with probability A ( i.e. min(1,exp(-(E(s')-E(s))/ temp) )
    and s_init with probability 1-A
    """
    delta_energy = energy_sprime - energy_s  # E(s')-E(s)
    exp_factor = np.exp(-delta_energy / temp)
    acceptance = min(
        1, exp_factor
    )  # for both QC case as well as uniform random strategy, the transition matrix Pij is symmetric!
    # coin_flip=np.random.choice([True, False], p=[acceptance, 1-acceptance])
    new_state = s_init
    if acceptance >= np.random.uniform(0, 1):
        new_state = s_prime
    return new_state


def classical_mcmc(
    N_hops: int,
    num_spins: int,
    initial_state: str,
    num_elems: int,
    model,
    return_last_n_states=500,
    return_both=False,
    temp=1,
):
    """
    Args:
    Nhops: Number of time you want to run mcmc
    num_spins: number of spins
    num_elems: 2**(num_spins)
    model:
    return_last_n_states: (int) Number of states in the end of the M.Chain you want to consider for prob distn (default value is last 500)
    return_both (default=False): If set to True, in addition to dict_count_return_lst_n_states, also returns 2 lists:
                                "list_after_transition: list of states s' obtained after transition step s->s' " and
                                "list_after_acceptance_step: list of states accepted after the accepance step".
    Returns:
    Last 'dict_count_return_last_n_states' elements of states so collected (default value=500). one can then deduce the distribution from it!
    """
    states = []
    # current_state=f'{np.random.randint(0,num_elems):0{num_spins}b}'# bin_next_state=f'{next_state:0{num_spins}b}'
    current_state = initial_state
    print("starting with: ", current_state)
    states.append(current_state)

    ## initialiiise observables
    # observable_dict = dict([ (elem, []) for elem in observables ])
    list_after_transition = []
    list_after_acceptance_step = []

    for i in tqdm(range(0, N_hops)):
        # get sprime
        s_prime = classical_transition(num_spins)
        list_after_transition.append(s_prime)
        # accept/reject s_prime
        energy_s = model.get_energy(current_state)
        energy_sprime = model.get_energy(s_prime)
        next_state = classical_loop_accepting_state(
            current_state, s_prime, energy_s, energy_sprime, temp=temp
        )
        current_state = next_state
        list_after_acceptance_step.append(current_state)
        states.append(current_state)
        # WE DON;T NEED TO DO THIS! # reinitiate
        # qc_s=initialise_qc(n_spins=num_spins, bitstring=current_state)

    # returns dictionary of occurences for last "return_last_n_states" states
    dict_count_return_last_n_states = Counter(states[-return_last_n_states:])

    if return_both:
        to_return = (
            dict_count_return_last_n_states,
            list_after_transition,
            list_after_acceptance_step,
        )
    else:
        to_return = dict_count_return_last_n_states

    return to_return


################################################################################################
##  Quantum Circuit for Quantum enhanced MCMC ##
################################################################################################


def initialise_qc(n_spins: int, bitstring: str):
    """
    Initialises a quantum circuit with n_spins number of qubits in a state defined by "bitstring"
    (Caution: Qiskit's indexing convention for qubits (order of tensor product) is different from the conventional textbook one!)
    """

    spins = QuantumRegister(n_spins, name="spin")
    creg_final = ClassicalRegister(n_spins, name="creg_f")
    qc_in = QuantumCircuit(spins, creg_final)

    len_str_in = len(bitstring)
    assert len_str_in == len(
        qc_in.qubits
    ), "len(bitstring) should be equal to number_of_qubits/spins"

    # print("qc_in.qubits: ", qc_in.qubits)
    where_x_gate = [
        qc_in.qubits[len_str_in - 1 - i]
        for i in range(0, len(bitstring))
        if bitstring[i] == "1"
    ]
    if len(where_x_gate) != 0:
        qc_in.x(where_x_gate)
    return qc_in


def fn_qc_h1(num_spins: int, gamma, alpha, h, delta_time):
    """
    Create a Quantum Circuit for time-evolution under
    hamiltonain H1 (described in the paper)

    Args:
    num_spins: number of spins in the model
    gamma: float
    alpha: float
    h: list of field at each site
    delta_time: total evolution time time/num_trotter_steps
    """
    a = gamma
    # print("a:",a)
    b_list = [-(1 - gamma) * alpha * hj for hj in h]
    list_unitaries = [
        UnitaryGate(
            HamiltonianGate(
                a * XGate().to_matrix() + b_list[j] * ZGate().to_matrix(),
                time=delta_time,
            ).to_matrix(),
            label=f"exp(-ia{j}X+b{j}Z)",
        )
        for j in range(0, num_spins)
    ]
    qc = QuantumCircuit(num_spins)
    for j in range(0, num_spins):
        qc.append(list_unitaries[j], [num_spins - 1 - j])
    qc.barrier()
    # print("qc is:"); print(qc.draw())
    return qc


def fn_qc_h2(J, alpha, gamma, delta_time=0.8):
    """
    Create a Quantum Circuit for time-evolution under
    hamiltonain H2 (described in the paper)

    Args:
    J: interaction matrix, interaction between different spins
    gamma: float
    alpha: float
    delta_time: (default=0.8, as suggested in the paper)total evolution time time/num_trotter_steps
    """
    num_spins = np.shape(J)[0]
    qc_for_evol_h2 = QuantumCircuit(num_spins)
    theta_list = [
        -2 * J[j, j + 1] * (1 - gamma) * alpha * delta_time
        for j in range(0, num_spins - 1)
    ]
    for j in range(0, num_spins - 1):
        qc_for_evol_h2.rzz(
            theta_list[j], qubit1=num_spins - 1 - j, qubit2=num_spins - 1 - (j + 1)
        )
    # print("qc for fn_qc_h2 is:"); print(qc_for_evol_h2.draw())
    return qc_for_evol_h2


def trottered_qc_for_transition(num_spins, qc_h1, qc_h2, num_trotter_steps):
    """returns a trotter circuit (evolution_under_h2 X evolution_under_h1)^(r-1) (evolution under h1)"""
    qc_combine = QuantumCircuit(num_spins)
    for i in range(0, num_trotter_steps - 1):
        qc_combine = qc_combine.compose(qc_h1)
        qc_combine = qc_combine.compose(qc_h2)
        qc_combine.barrier()
    qc_combine = qc_combine.compose(qc_h1)
    # print("trotter ckt:"); print(qc_combine.draw())
    return qc_combine


def combine_2_qc(init_qc: QuantumCircuit, trottered_qc: QuantumCircuit):
    """Function to combine 2 quantum ckts of compatible size.
    In this project, it is used to combine initialised quantum ckt and quant ckt meant for time evolution
    """
    num_spins = len(init_qc.qubits)
    qc = QuantumCircuit(num_spins, num_spins)
    qc = qc.compose(init_qc)
    qc.barrier()
    qc = qc.compose(trottered_qc)
    return qc


################################################################################################
##  Quantum MCMC routines ##
################################################################################################


def run_qc_quantum_step(
    qc_initialised_to_s: QuantumCircuit, model: IsingEnergyFunction, alpha, n_spins: int
) -> str:

    """
    Takes in a qc initialized to some state "s". After performing unitary evolution U=exp(-iHt)
    , circuit is measured once. Function returns the bitstring s', the measured state .

    Args:
    qc_initialised_to_s:
    model:
    alpha:
    n_spins:
    """

    h = model.get_h()
    J = model.get_J()

    # init_qc=initialise_qc(n_spins=n_spins, bitstring='1'*n_spins)
    gamma = np.round(np.random.uniform(0.25, 0.6), decimals=2)
    time = np.random.choice(list(range(2, 12)))  # earlier I had [2,20]
    delta_time = 0.8
    num_trotter_steps = int(np.floor((time / delta_time)))
    # print(f"gamma:{gamma}, time: {time}, delta_time: {delta_time}, num_trotter_steps:{num_trotter_steps}")
    # print(f"num troter steps: {num_trotter_steps}")
    qc_evol_h1 = fn_qc_h1(n_spins, gamma, alpha, h, delta_time)
    qc_evol_h2 = fn_qc_h2(J, alpha, gamma, delta_time=delta_time)
    trotter_ckt = trottered_qc_for_transition(
        n_spins, qc_evol_h1, qc_evol_h2, num_trotter_steps=num_trotter_steps
    )
    qc_for_mcmc = combine_2_qc(qc_initialised_to_s, trotter_ckt)

    # run the circuit
    num_shots = 1
    quantum_registers_for_spins = qc_for_mcmc.qregs[0]
    classical_register = qc_for_mcmc.cregs[0]
    qc_for_mcmc.measure(quantum_registers_for_spins, classical_register)
    # print("qc_for_mcmc: ")
    # print( qc_for_mcmc.draw())
    state_obtained_dict = (
        execute(qc_for_mcmc, shots=num_shots, backend=qsm).result().get_counts()
    )
    state_obtained = list(state_obtained_dict.keys())[
        0
    ]  # since there is only one element
    return state_obtained


def quantum_enhanced_mcmc(
    N_hops: int,
    num_spins: int,
    initial_state: str,
    num_elems: int,
    model: IsingEnergyFunction,
    alpha,
    return_last_n_states=500,
    return_both=False,
    temp=1,
):
    """
    version 0.2
    Args:
    Nhops: Number of time you want to run mcmc
    num_spins: number of spins
    num_elems: 2**(num_spins)
    model:
    alpha:
    return_last_n_states:
    return_both:
    temp:

    Returns:
    Last 'return_last_n_states' elements of states so collected (default value=500). one can then deduce the distribution from it!
    """
    states = []
    print("starting with: ", initial_state)

    ## initialise quantum circuit to current_state
    qc_s = initialise_qc(n_spins=num_spins, bitstring=initial_state)
    current_state = initial_state
    states.append(current_state)
    ## intialise observables
    list_after_transition = []
    list_after_acceptance_step = []

    for i in tqdm(range(0, N_hops)):
        # print("i: ", i)
        # get sprime
        s_prime = run_qc_quantum_step(
            qc_initialised_to_s=qc_s, model=model, alpha=alpha, n_spins=num_spins
        )
        list_after_transition.append(s_prime)
        # accept/reject s_prime
        energy_s = model.get_energy(current_state)
        energy_sprime = model.get_energy(s_prime)
        next_state = classical_loop_accepting_state(
            current_state, s_prime, energy_s, energy_sprime, temp=temp
        )
        current_state = next_state
        list_after_acceptance_step.append(current_state)
        states.append(current_state)
        ## reinitiate
        qc_s = initialise_qc(n_spins=num_spins, bitstring=current_state)

    dict_count_return_last_n_states = Counter(
        states[-return_last_n_states:]
    )  # dictionary of occurences for last "return_last_n_states" states

    if return_both:
        to_return = (
            dict_count_return_last_n_states,
            list_after_transition,
            list_after_acceptance_step,
        )
    else:
        to_return = dict_count_return_last_n_states

    return to_return


###################################
# Some New Helper functions
###################################
def states(num_spins: int) -> list:
    """
    Returns all possible binary strings of length n=num_spins

    Args:
    num_spins: n length of the bitstring
    Returns:
    possible_states= list of all possible binary strings of length num_spins
    """
    num_possible_states = 2 ** (num_spins)
    possible_states = [f"{k:0{num_spins}b}" for k in range(0, num_possible_states)]
    return possible_states


def magnetization_of_state(bitstring: str) -> float:
    """
    Args:
    bitstring: for eg: '010'
    Returns:
    magnetization for the given bitstring
    """
    array = np.array(list(bitstring))
    num_times_one = np.count_nonzero(array == "1")
    num_times_zero = len(array) - num_times_one
    magnetization = num_times_one - num_times_zero
    return magnetization


def dict_magnetization_of_all_states(list_all_possible_states: list) -> dict:
    """
    Returns magnetization for all unique states

    Args:
    list_all_possible_states
    Returns:
    dict_magnetization={state(str): magnetization_value}
    """
    list_mag_vals = [
        magnetization_of_state(state) for state in list_all_possible_states
    ]
    dict_magnetization = dict(zip(list_all_possible_states, list_mag_vals))
    # print("dict_magnetization:"); print(dict_magnetization)
    return dict_magnetization


def value_sorted_dict(dict_in, reverse=False):
    """Sort the dictionary in ascending or descending(if reverse=True) order of values"""
    sorted_dict = {
        k: v
        for k, v in sorted(dict_in.items(), key=lambda item: item[1], reverse=reverse)
    }
    return sorted_dict


## enter samples, get normalised distn
def get_distn(list_of_samples: list) -> dict:
    """
    Returns the dictionary of distn for input list_of_samples
    """
    len_list = len(list_of_samples)
    temp_dict = Counter(list_of_samples)
    temp_prob_list = np.array(list(temp_dict.values())) * (1.0 / len_list)
    dict_to_return = dict(zip(list(temp_dict.keys()), temp_prob_list))
    return dict_to_return


## Average
def avg(dict_probabilities: dict, dict_observable_val_at_states: dict):
    """
    new version:
    Returns average of any observable of interest

    Args:
    dict_probabilities= {state: probability}
    dict_observable_val_at_states={state (same as that of dict_probabilities): observable's value at that state}

    Returns:
    avg
    """
    len_dict = len(dict_probabilities)
    temp_list = [
        dict_probabilities[j] * dict_observable_val_at_states[j]
        for j in (list(dict_probabilities.keys()))
    ]
    avg = np.sum(
        temp_list
    )  # earlier I had np.mean here , which is wrong (obviously! duh!)
    return avg


### function to get running average of magnetization
def running_avg_magnetization(list_states_mcmc: list):
    """
    Returns the running average magnetization

    Args:
    list_states_mcmc= List of states aceepted after each MCMC step
    """
    len_iters_mcmc = len(list_states_mcmc)
    running_avg_mag = {}
    for i in tqdm(range(1, len_iters_mcmc)):
        temp_list = list_states_mcmc[:i]  # [:i]
        temp_prob = get_distn(temp_list)
        dict_mag_states_in_temp_prob = dict_magnetization_of_all_states(temp_list)
        running_avg_mag[i] = avg(temp_prob, dict_mag_states_in_temp_prob)
    return running_avg_mag


def running_avg_magnetization_as_list(list_states_mcmc: list):
    """
    Returns the running average magnetization

    Args:
    list_states_mcmc= List of states aceepted after each MCMC step
    """
    list_of_strings = list_states_mcmc
    list_of_lists = (
        np.array([list(int(s) for s in bitstring) for bitstring in list_of_strings]) * 2
        - 1
    )
    return np.array(
        [
            np.mean(np.sum(list_of_lists, axis=1)[:ii])
            for ii in range(1, len(list_states_mcmc) + 1)
        ]
    )


## Plotting related
def plot_dict_of_running_avg_observable(
    dict_running_avg: dict, observable_legend_label: str
):
    plt.plot(
        list(dict_running_avg.keys()),
        list(dict_running_avg.values()),
        "-",
        label=observable_legend_label,
    )
    plt.xlabel("MCMC iterations")


def plot_bargraph_desc_order(
    desc_val_order_dict_in: dict,
    label: str,
    normalise_complete_data: bool = False,
    plot_first_few: int = 0,
):
    width = 1.0
    list_keys = list(desc_val_order_dict_in.keys())
    list_vals = list(desc_val_order_dict_in.values())
    if normalise_complete_data:
        list_vals = np.divide(
            list_vals, sum(list_vals)
        )  # np.divide(list(vals), sum(vals))
    if plot_first_few != 0:
        plt.bar(list_keys[0:plot_first_few], list_vals[0:plot_first_few], label=label)
    else:
        plt.bar(list_keys, list_vals, label=label)
    plt.xticks(rotation=90)


def plot_multiple_bargraphs(
    list_of_dicts: list,
    list_labels: list,
    list_normalise: list,
    plot_first_few,
    sort_desc=False,
    sort_asc=False,
    figsize=(15, 7),
):
    list_keys = list(list_of_dicts[0].keys())
    dict_data = {}
    for i in range(0, len(list_labels)):
        # list_vals=[list_of_dicts[i][j] for j in list_keys if j in list(list_of_dicts[i].keys()) else 0] #list(list_of_dicts[i].values())
        list_vals = [
            list_of_dicts[i][j] if j in list(list_of_dicts[i].keys()) else 0
            for j in list_keys
        ]
        if list_normalise[i]:
            list_vals = np.divide(list_vals, sum(list_vals))
        dict_data[list_labels[i]] = list_vals
    df = pd.DataFrame(dict_data, index=list_keys)
    if sort_desc:
        df_sorted_desc = df.sort_values(list_labels[0], ascending=False)
        df_sorted_desc[:plot_first_few].plot.bar(rot=90, figsize=figsize)
    elif sort_asc:
        df_sorted_asc = df.sort_values(list_labels[0], ascending=True)
        df_sorted_asc[:plot_first_few].plot.bar(rot=90, figsize=figsize)
    elif sort_desc == False and sort_asc == False:
        df[:plot_first_few].plot.bar(rot=90, figsize=figsize)


## Hamming distance related
# hamming distance between 2 strings
def hamming_dist(str1, str2):
    i = 0
    count = 0
    while i < len(str1):
        if str1[i] != str2[i]:
            count += 1
        i += 1
    return count


def hamming_dist_related_counts(
    num_spins: int, sprime_each_iter: list, states_accepted_each_iter: list
):

    dict_counts_states_hamming_dist = dict(
        zip(list(range(0, num_spins + 1)), [0] * (num_spins + 1))
    )
    ham_dist_s_and_sprime = np.array(
        [
            hamming_dist(states_accepted_each_iter[j], sprime_each_iter[j + 1])
            for j in range(0, len(states_accepted_each_iter) - 1)
        ]
    )
    for k in list(dict_counts_states_hamming_dist.keys()):
        dict_counts_states_hamming_dist[k] = np.count_nonzero(
            ham_dist_s_and_sprime == k
        )

    assert (
        sum(list(dict_counts_states_hamming_dist.values())) == len(sprime_each_iter) - 1
    )
    return dict_counts_states_hamming_dist


def energy_difference_related_counts(
    num_spins, sprime_each_iter: list, states_accepted_each_iter: list, model_in
):

    energy_diff_s_and_sprime = np.array(
        [
            abs(
                model_in.get_energy(sprime_each_iter[j])
                - model_in.get_energy(states_accepted_each_iter[j + 1])
            )
            for j in range(0, len(sprime_each_iter) - 1)
        ]
    )
    return energy_diff_s_and_sprime


# function to create dict for number of times states sprime were not accepted in MCMC iterations
def fn_numtimes_bitstring_not_accepted(list_after_trsn, list_after_accept, bitstring):

    where_sprime_is_bitstr = list(np.where(np.array(list_after_trsn) == bitstring)[0])
    where_bitstr_not_accepted = [
        k for k in where_sprime_is_bitstr if list_after_accept[k] != bitstring
    ]
    numtimes_sprime_is_bitstring = len(where_sprime_is_bitstr)
    numtimes_bitstring_not_accepted = len(where_bitstr_not_accepted)
    return numtimes_bitstring_not_accepted, numtimes_sprime_is_bitstring


def fn_states_not_accepted(
    list_states: list, list_after_trsn: list, list_after_accept: list
):
    list_numtimes_state_not_accepted = [
        fn_numtimes_bitstring_not_accepted(list_after_trsn, list_after_accept, k)[0]
        for k in list_states
    ]
    list_numtimes_sprime_is_state = [
        fn_numtimes_bitstring_not_accepted(list_after_trsn, list_after_accept, k)[1]
        for k in list_states
    ]
    dict_numtimes_states_not_accepted = dict(
        zip(list_states, list_numtimes_state_not_accepted)
    )
    dict_numtimes_sprime_is_state = dict(
        zip(list_states, list_numtimes_sprime_is_state)
    )
    return dict_numtimes_states_not_accepted, dict_numtimes_sprime_is_state
