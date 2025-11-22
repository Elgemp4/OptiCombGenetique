from utils import fobj
import numpy as np

class Solution:
    def __init__(self, W: np.ndarray, H: np.ndarray):
        self.W = W
        self.H = H
        self.score = None

    def get_score(self) -> int:
        return self.score

    def compute_score(self, X):
        self.score = fobj(X,self.W,self.H)

    def get_W(self) -> np.ndarray:
        return self.W

    def get_H(self) -> np.ndarray:
       return self.H