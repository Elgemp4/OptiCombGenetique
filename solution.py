from utils import fobj

class Solution:
    def __init__(self, solution):
        self.solution = solution
        self.score = fobj(solution)

    def get_score(self):
        return self.score

    def get_solution(self):
        return self.solution