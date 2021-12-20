from typing import List, Dict

import numpy as np
from ortools.sat.python.cp_model import CpSolverSolutionCallback

from utils import Couple_t, Matching_t


class MySolutionCallback(CpSolverSolutionCallback):
    solver = None

    def set_solver(self, solver):
        self.solver = solver


class CompositeSolutionCallback(MySolutionCallback):
    def __init__(self, callbacks: List[MySolutionCallback]):
        super().__init__()
        self.callbacks = callbacks
        for callback in self.callbacks:
            callback.set_solver(self)

    def on_solution_callback(self):
        try:
            for callback in self.callbacks:
                callback.OnSolutionCallback()
        except Exception as e:
            print("composite:", e)


class MatchSolutionCounter(MySolutionCallback):
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
            self._soln_cnts[mw] += self.solver.Value(x_mw)

    def n_solutions(self):
        return self._total_solns

    def counts(self):
        return self._soln_cnts


class SolutionScorerCallback(MySolutionCallback):
    def __init__(self, variables: Dict):
        super().__init__()
        self.variable_map = variables
        self.keys = sorted(self.variable_map.keys())
        self.variables = [self.variable_map[mw] for mw in self.keys]
        self.solutions = []

        self._has_comp_distr = False
        self._expected_n_sols = []
        self._expected_lights = []

    def on_solution_callback(self):
        self.solutions.append([self.solver.Value(x_mw) for x_mw in self.variables])

    def by_entropy(self):
        solutions = np.array(self.solutions)
        counts = solutions.sum(axis=0)
        probs = counts / counts.sum()  # prob[i] is probability couple_i is in solution
        entropies = np.log(probs)
        return entropies

    def scores(self):
        if self._has_comp_distr:
            return self._expected_n_sols
        self._has_comp_distr = True
        sols = np.array(self.solutions)
        n_sols = len(self.solutions)

        # lights[i,j] gives us the number of lights sol. i gets if sol. j is perfect
        lights = np.dot(sols, sols.T)
        self._expected_lights = np.mean(lights, axis=1)

        # gives us number of solutions in which i gets k lights
        n_sols_by_lights = np.apply_along_axis(np.bincount, 1, lights)

        # each sol's expected num lights
        self._expected_n_sols = (n_sols_by_lights * n_sols_by_lights / n_sols).sum(axis=1)
        return self._expected_n_sols

    def lights(self):
        if self._has_comp_distr:
            return self._expected_lights
        self.scores()
        return self._expected_lights

    def min_expected_matching(self) -> List[Couple_t]:
        # best solutions is one which has the minimum number of expected solutions
        scores = self.scores()
        argmin_exp = np.argmin(scores)
        s = self.solutions[argmin_exp]
        print(f"Optimal is matching {argmin_exp} with expected num sols remaining = {scores[argmin_exp]}")
        return [mw for i, mw in enumerate(self.keys) if s[i] == 1]

    def evaluate_matching(self, matching: Matching_t) -> float:
        m = np.array(list(map(lambda mw: int(mw in matching), self.keys)))
        n_sols = len(self.solutions)
        lights = np.dot(self.solutions, m)
        n_sols_by_lights = np.bincount(lights)
        expected_n_sols = np.dot(n_sols_by_lights, n_sols_by_lights / n_sols).sum()
        return expected_n_sols

    def expected_lights(self, matching: Matching_t):
        m = np.array(list(map(lambda mw: int(mw in matching), self.keys)))
        lights = np.dot(self.solutions, m)
        return np.mean(lights)
