import pdb


class Subgraph():

    """
    def __init__(self, edges, num_nodes, inner, outer):
        self.size = num_nodes
        self.inner = inner
        self.outer = outer
        self.edge_size = len(edges)

        self.M = {}

        for (i, j, w) in edges:
            if i not in self.M:
                self.M[i] = {}

            self.M[i][j] = w

            if j not in self.M:
                self.M[j] = {}

            self.M[j][i] = w


    def get_seed_score(self, seed):
        score = 0.0

        visited = set()
        queue = [seed]
        visited.add(seed)

        while len(queue) > 0:
            i = queue.pop()
            if i in self.inner:
                score += 1
            else:
                score -= 1

            if i not in self.M:
                continue
            
            for j in self.M[i]:
                if j not in visited:
                    visited.add(j)
                    queue.append(j)

        return score
    """

    def __init__(self, edges, inner, outer):
        self.edges = edges
        self.inner = inner
        self.outer = outer

    # Given a seed returns the score that seed would have gotten.
    def get_seed_score(self, seeds):

        score = 0.0

        visited = set()
        queue = list(seeds)
        for s in seeds:
            if s in visited:
                pdb.set_trace()
            visited.add(s)

        while len(queue) > 0:
            i = queue.pop()

            if i in self.inner:
                score += 1
            else:
                score -= 1

            for j in self.edges[i]:
                if j not in visited:
                    visited.add(j)
                    queue.append(j)

        return score
    
