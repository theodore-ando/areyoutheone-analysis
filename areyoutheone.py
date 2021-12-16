from typing import List

import pandas as pd
from ortools.sat.python.cp_model import CpModel, CpSolver
from pandas.io.formats.style import Styler

from solutioncallbacks import MatchSolutionCounter, CompositeSolutionCallback, SolutionScorerCallback
from utils import Person_t, Couple_t, Matching_t


class AreYouTheOne:
    def __init__(self, men: List[Person_t], women: List[Person_t]):
        if len(men) != len(women):
            raise ValueError("Number of men and women must be same")
        if len(men) == 0:
            raise ValueError("Must have positive number of people")
        self.men = men
        self.women = women
        self.model = CpModel()
        self.solver = CpSolver()
        self.solver.parameters.enumerate_all_solutions = True

        # x_(m,w) is 1 if the couple (m,w) is a match, 0 otherwise
        self.x = {
            (m, w): self.model.NewBoolVar(f"Match({m}, {w})")
            for m in men
            for w in women
        }
        self.counter = MatchSolutionCounter(self.x)
        self.scorer = SolutionScorerCallback(self.x)

        # ensure each woman only has 1 match
        for w in women:
            self.model.Add(1 == sum(self.x[(m, w)] for m in men))

        # ensure each man only has 1 match
        for m in men:
            self.model.Add(1 == sum(self.x[(m, w)] for w in women))

    def add_matching_ceremony(self, n_perfect: int, matches: List[Couple_t]):
        self.model.Add(n_perfect == sum(self.x[mw] for mw in matches))

    def add_truth_booth(self, m: str, w: str, perfect: bool):
        self.model.Add(self.x[(m, w)] == int(perfect))

    def solve(self, solution_callback=None, **kwargs) -> MatchSolutionCounter:
        self.counter = MatchSolutionCounter(self.x, **kwargs)
        self.scorer = SolutionScorerCallback(self.x)
        if solution_callback is not None:
            solution_callback = CompositeSolutionCallback([self.counter, self.scorer, solution_callback])
        else:
            solution_callback = CompositeSolutionCallback([self.counter, self.scorer])
        self.solver.Solve(self.model, solution_callback)
        return self.counter

    def minmax_truth_booth(self, couples: List[Couple_t]) -> Couple_t:
        """Pick the couple that provides the min max number of possible remaining solutions"""
        max_solutions = {}
        N = self.counter.n_solutions()
        for couple in couples:
            n0 = N - self.counter.counts()[couple]
            n1 = self.counter.counts()[couple]
            max_solutions[couple] = max(n0, n1)
        print(max_solutions)
        return min(couples, key=lambda c: max_solutions[c])

    def min_expected_truth_booth(self, couples: List[Couple_t]) -> Couple_t:
        """Pick the couple that provides the min expected number of remaining solutions"""
        exp_solns = {}
        N = self.counter.n_solutions()
        for couple in couples:
            p0 = 1 - self.counter.counts()[couple] / N
            p1 = self.counter.counts()[couple] / N

            n0 = N - self.counter.counts()[couple]
            n1 = self.counter.counts()[couple]
            exp_solns[couple] = p0 * n0 + p1 * n1
        print(exp_solns)
        # isn't this actually just the couple with p closest to 0.5?
        return min(couples, key=lambda c: exp_solns[c])

    def min_expected_matching(self) -> Matching_t:
        """Pick the matching that provides the min expected number of remaining solutions"""
        return self.scorer.min_expected_matching()

    def evaluate_matching(self, matching: Matching_t) -> float:
        """Pick the matching that provides the min expected number of remaining solutions"""
        return self.scorer.evaluate_matching(matching)

    def display(self) -> Styler:
        counts = self.counter.counts()
        df = pd.DataFrame(
            data=[[counts[(m, w)] for w in self.women] for m in self.men],
            index=self.men,
            columns=self.women
        )
        return df.style.background_gradient(axis=None) \
            .set_caption(f"Number of Matches in {self.counter.n_solutions()} Solutions") \
            .set_table_styles([{
                'selector': 'caption',
                'props': [
                    ('color', 'black'),
                    ('font-size', '16pt'),
                    ('caption-side', 'center')
                ]
            }])
