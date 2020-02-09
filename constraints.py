from typing import List

class Constraint():
    """Constraints loaded from a file."""

    def __init__(self, fname):
        """
        Construct a Constraint object from a constraints file

        :param fname: Name of the file to read the Constraint from (string)
        """
        with open(fname, "r") as f:
            lines = f.readlines()
        # Parse the dimension from the first line
        self.n_dim = int(lines[0])
        # Parse the example from the second line
        self.example = [float(x) for x in lines[1].split(" ")[0:self.n_dim]]

        # Run through the rest of the lines and compile the constraints
        self.exprs, self.num_exprs = [], []
        for i in range(2, len(lines)):
            # support comments in the first line
            if lines[i][0] == "#":
                continue
            self.exprs.append(compile(lines[i], "<string>", "eval"))
            self.num_exprs.append(compile(lines[i].strip(" >= 0 \n"), "<string>", "eval"))
        return

    def get_example(self):
        """Get the example feasible vector"""
        return self.example

    def get_ndim(self):
        """Get the dimension of the space on which the constraints are defined"""
        return self.n_dim

    def apply(self, x):
        """
        Apply the constraints to a vector, returning True only if all are satisfied

        :param x: list or array on which to evaluate the constraints
        """
        for expr in self.exprs:
            if not eval(expr):
                 return False
        return eval(expr)

    def eval_constraints(self, x: np.Array[float]) -> np.Array[float]:
        """
        Evaluate g(x) for each g(x) >= 0 constraint, returning the values of every evaluation as a list
        :param x: list on which to evaluate g(x)
        :return: list of g(x)
        """
        results = []
        for num_expr in self.num_exprs:
            results.append(eval(num_expr))
        return np.array(results)
