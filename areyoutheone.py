from math import sqrt, ceil
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle
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
        self.truth_booths = {}
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
        self.truth_booths[(self.men.index(m), self.women.index(w))] = int(perfect)
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

    def expected_lights(self, matching: Matching_t) -> float:
        """Compute the expected number of lights that this matching will get given current model state"""
        return self.scorer.expected_lights(matching)

    def display(self):
        counts = self.counter.counts()
        df = pd.DataFrame(
            data=[[counts[(m, w)] for w in self.women] for m in self.men],
            index=self.men,
            columns=self.women
        )

        mask = np.zeros((len(self.men), len(self.women)))
        for (i, j), b in self.truth_booths.items():
            mask[i, j] = b
        f, ax = plt.subplots(figsize=(9, 6))
        ax.set_title(f"Number of Matches in {self.counter.n_solutions()} Solutions")
        sns.heatmap(df, annot=True, ax=ax, fmt='d', cmap="Blues", mask=mask)

        # apply green / red fill in boxes where we've gotten a TB
        for (i, j), b in self.truth_booths.items():
            color = 'green' if b else 'red'
            # note that Rectangle loc is (x,y) not (row,col) hence we flip i,j
            ax.add_patch(Rectangle((j, i), 1, 1, color=color, fill=True, lw=0, alpha=0.5))

        # apply green / red outline in boxes where we've inferred a sure outcome
        loc_nonmatches = np.where(df == 0)
        loc_matches = np.where(df == self.counter.n_solutions())
        tb_match_rows = {i for (i, j), b in self.truth_booths.items() if b == 1}
        tb_match_cols = {j for (i, j), b in self.truth_booths.items() if b == 1}
        inferred_nonmatches = {
                (i, j)
                for (i, j) in zip(*loc_nonmatches)
                if (i not in tb_match_rows) and (j not in tb_match_cols) and ((i, j) not in self.truth_booths)
        }
        for (i, j) in inferred_nonmatches:
            # note that Rectangle loc is (x,y) not (row,col) hence we flip i,j
            ax.add_patch(Rectangle((j, i), 1, 1, color='red', fill=False, lw=1, alpha=0.5))
        for (i, j) in zip(*loc_matches):
            if (i, j) not in self.truth_booths:
                # note that Rectangle loc is (x,y) not (row,col) hence we flip i,j
                ax.add_patch(Rectangle((j, i), 1, 1, color='green', fill=False, lw=1))

    def display_distributions(self, matching: Matching_t):
        expected_n_sols = self.evaluate_matching(matching)
        expected_n_lights = self.expected_lights(matching)

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))

        # plot the distribution of expected number of solutions remaining, along with their choice
        ax0.set_title('Distribution of E[# of solutions]')
        ax0.set_ylabel('frequency')
        ax0.set_xlabel(r'$\mathbb{E}$[# of solutions]', )
        sns.histplot(self.scorer.scores(), ax=ax0)
        ax0.axvline(expected_n_sols, color='r')

        # plot the distribution of expected number of lights, along with their choice
        ax1.set_title('Distribution of E[\# of lights]')
        ax1.set_ylabel('frequency')
        ax1.set_xlabel('E[# of lights]')
        sns.histplot(self.scorer.lights(), ax=ax1)
        ax1.axvline(expected_n_lights, color='r')
