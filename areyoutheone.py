from typing import List, Tuple

import pandas as pd
from ortools.sat.python.cp_model import CpModel, CpSolver

from solutioncallbacks import MatchSolutionCounter


class AreYouTheOne:
    def __init__(self, men: List[str], women: List[str]):
        if len(men) != len(women):
            raise ValueError("Number of men and women must be same")
        if len(men) == 0:
            raise ValueError("Must have positive number of people")
        self.men = men
        self.women = women
        self.model = CpModel()
        self.solver = CpSolver()
        self.solver.parameters.enumerate_all_solutions = True
        self.counter: MatchSolutionCounter = None
        self.x = {
            (m, w): self.model.NewBoolVar(f"Match({m}, {w})")
            for m in men
            for w in women
        }

        # ensure each woman only has 1 match
        for w in women:
            self.model.Add(1 == sum(self.x[(m, w)] for m in men))

        # ensure each man only has 1 match
        for m in men:
            self.model.Add(1 == sum(self.x[(m, w)] for w in women))

    def add_matching_ceremony(self, n_perfect: int, matches: List[Tuple[str, str]]):
        self.model.Add(n_perfect == sum(self.x[mw] for mw in matches))

    def add_truth_booth(self, m: str, w: str, perfect: bool):
        self.model.Add(self.x[(m, w)] == int(perfect))

    def solve(self, **kwargs) -> MatchSolutionCounter:
        self.counter = MatchSolutionCounter(self.x, **kwargs)

        self.solver.Solve(self.model, self.counter)
        return self.counter

    def display(self):
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
