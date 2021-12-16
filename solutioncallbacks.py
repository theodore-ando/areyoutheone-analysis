from typing import List, Dict

import numpy as np
from ortools.sat.python.cp_model import CpSolverSolutionCallback

from utils import Couple_t


class CompositeSolutionCallback(CpSolverSolutionCallback):
    def __init__(self, callbacks: List[CpSolverSolutionCallback]):
        super().__init__()
        self.callbacks = callbacks

    def on_solution_callback(self):
        for callback in self.callbacks:
            callback.OnSolutionCallback()


class MatchSolutionCounter(CpSolverSolutionCallback):
    def __init__(self, variables, print_every=100):
        super().__init__()
        self._variables = variables
        self._print_every = print_every
        self._soln_cnts = {mw: 0 for mw, x_mw in variables.items()}
        self._total_solns = 0

    def on_solution_callback(self):
        self._total_solns += 1
        if self._total_solns % self._print_every == 0:
            print(self._total_solns)
        for mw, x_mw in self._variables.items():
            self._soln_cnts[mw] += self.Value(x_mw)

    def n_solutions(self):
        return self._total_solns

    def counts(self):
        return self._soln_cnts


class SolutionScorerCallback(CpSolverSolutionCallback):
    def __init__(self, variables: Dict):
        super().__init__()
        self.variable_map = variables
        self.keys = sorted(self.variable_map.keys())
        self.variables = [self.variable_map[mw] for mw in self.keys]
        self.solutions = []

    def on_solution_callback(self):
        self.solutions.append([self.Value(x_mw) for mw, x_mw in self.variable_map.items()])

    def by_entropy(self):
        solutions = np.array(self.solutions)
        counts = solutions.sum(axis=0)
        probs = counts / counts.sum()  # prob[i] is probability couple_i is in solution
        entropies = np.log(probs)
        return entropies

    def min_expected_matching(self) -> List[Couple_t]:
        sols = np.array(self.solutions)
        n_sols = len(self.solutions)

        # lights[i,j] gives us the number of lights sol. i gets if sol. j is perfect
        lights = np.dot(sols, sols.T)

        # gives us number of solutions in which i gets k lights
        n_sols_by_lights = np.apply_along_axis(np.bincount, 1, lights)

        # each sol's expected num lights
        expected_n_sols = (n_sols_by_lights * n_sols_by_lights / n_sols).sum(axis=1)

        # best solutions is one which has the minimum number of expected solutions
        argmin_exp = np.argmin(expected_n_sols)
        s = self.solutions[argmin_exp]
        print(f"Optimal is matching {argmin_exp} with expected num sols remaining = {expected_n_sols[argmin_exp]}")
        return [mw for i, mw in enumerate(self.keys) if s[i] == 1]

    def evaluate_matching(self, matching: List[Couple_t]) -> float:
        m = np.array(map(lambda mw: int(mw in matching), self.keys))
        n_sols = len(self.solutions)
        lights = np.dot(self.solutions, m)
        n_sols_by_lights = np.bincount(lights)
        expected_n_sols = np.dot(n_sols_by_lights, n_sols_by_lights / n_sols).sum()
        return expected_n_sols
