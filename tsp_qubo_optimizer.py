import numpy as np
import random, math
from qubots.base_optimizer import BaseOptimizer

class TSPQUBOOptimizer(BaseOptimizer):
    """
    QUBO-based optimizer for the Traveling Salesman Problem.

    The decision variables x[i, t] (flattened to a vector) indicate whether city i is visited at
    position t in the tour. The QUBO is constructed with an objective term (the tour cost) and
    penalty terms to enforce that each city is visited exactly once and that each tour position is
    occupied by exactly one city.
    """

    def __init__(self, time_limit=300, num_iterations=10000, initial_temperature=10.0, cooling_rate=0.999, penalty_multiplier=None):
        self.time_limit = time_limit
        self.num_iterations = num_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.penalty_multiplier = penalty_multiplier  # if None, set to max_distance * n

    def optimize(self, problem, initial_solution=None, **kwargs):
        # Number of cities (and positions)
        n = problem.nb_cities
        N = n
        m = N * N  # total number of binary variables

        # Get the distance matrix
        dist_matrix = np.array(problem.dist_matrix)  # shape (n, n)
        max_distance = np.max(dist_matrix)
        # Set penalty parameters (A for "each city once" and B for "each position once")
        A = self.penalty_multiplier if self.penalty_multiplier is not None else max_distance * n
        B = A  # we use the same penalty for both constraints

        # Build the QUBO matrix as a dictionary with keys (p,q) for p <= q.
        Q = {}

        def add_to_Q(p, q, value):
            if p > q:
                p, q = q, p
            Q[(p, q)] = Q.get((p, q), 0) + value

        # --- Objective term: tour cost ---
        # For each position t, and for each pair of cities (i, j), add:
        # d(i,j) * x[i,t] * x[j, (t+1) mod n]
        for t in range(N):
            t_next = (t + 1) % N
            for i in range(n):
                for j in range(n):
                    p = i * N + t
                    q = j * N + t_next
                    add_to_Q(p, q, dist_matrix[i, j])

        # --- Constraint 1: Each city is visited exactly once ---
        # For each city i, add A*(sum_t x[i,t] - 1)^2.
        for i in range(n):
            for t1 in range(N):
                p = i * N + t1
                # Diagonal contribution: from -2*x + x => net -1 times A for each x.
                add_to_Q(p, p, -A)
                for t2 in range(t1 + 1, N):
                    q = i * N + t2
                    add_to_Q(p, q, 2 * A)

        # --- Constraint 2: Each position is occupied by exactly one city ---
        # For each position t, add B*(sum_i x[i,t] - 1)^2.
        for t in range(N):
            for i1 in range(n):
                p = i1 * N + t
                add_to_Q(p, p, -B)
                for i2 in range(i1 + 1, n):
                    q = i2 * N + t
                    add_to_Q(p, q, 2 * B)

        # --- Solve the QUBO using simulated annealing ---
        # Initialize with a random binary vector of length m.
        x = np.random.randint(0, 2, size=m)
        current_cost = self.qubo_cost(x, Q)
        best_x = x.copy()
        best_cost = current_cost

        T = self.initial_temperature
        for it in range(self.num_iterations):
            # Pick a random variable to flip.
            p = random.randint(0, m - 1)
            x_new = x.copy()
            x_new[p] = 1 - x_new[p]
            new_cost = self.qubo_cost(x_new, Q)
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                x = x_new
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
            T *= self.cooling_rate
            if T < 1e-8:
                break

        # --- Decode the binary vector into a tour ---
        # Reshape best_x into an (n x n) assignment matrix.
        X = best_x.reshape((n, n))
        tour = []
        for t in range(n):
            col = X[:, t]
            # If exactly one city is assigned at position t, use it; otherwise take argmax.
            if np.sum(col) == 1:
                i = int(np.where(col == 1)[0][0])
            else:
                i = int(np.argmax(col))
            tour.append(i)
        # Optionally, rotate the tour so that city 0 is first.
        if 0 in tour:
            idx = tour.index(0)
            tour = tour[idx:] + tour[:idx]

        # Use the problem's own evaluation function.
        cost = problem.evaluate_solution(tour)
        return tour, cost

    def qubo_cost(self, x, Q):
        """Compute the QUBO cost given binary vector x and QUBO dictionary Q."""
        cost = 0
        # Q is stored for p <= q. The full quadratic form is:
        # cost = sum_p Q[p,p]*x[p] + 2 * sum_{p < q} Q[p,q]*x[p]*x[q]
        for (p, q), coeff in Q.items():
            if p == q:
                cost += coeff * x[p]
            else:
                cost += 2 * coeff * x[p] * x[q]
        return cost
