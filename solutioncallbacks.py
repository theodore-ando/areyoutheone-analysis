from ortools.sat.python import cp_model


class MatchSolutionCounter(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables, print_every=100):
        cp_model.CpSolverSolutionCallback.__init__(self)
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
