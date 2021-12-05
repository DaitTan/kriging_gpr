
class callCounter:
    def __init__(self,func):
        self.func = func
        self.callCount = 0
    def __call__(self, *args, **kwargs):
        self.callCount = self.callCount + 1
        return self.func(*args, **kwargs)


# def test_function(X):  ##CHANGE
#     # return (X[0]**2 + X[1] - 11)**2 + (X[1]**2 + X[0] - 7)**2 - 40 # Himmelblau's
#     # return (100 * (X[1] - X[0] **2)**2 + ((1 - X[0])**2)) - 20 # Rosenbrock
#     return (1 + (X[0] + X[1] + 1) ** 2 * (
#                 19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
#                        30 + (2 * X[0] - 3 * X[1]) ** 2 * (
#                            18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50
