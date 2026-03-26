class Multiplier:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor

    def __repr__(self):
        return f"Multiplier({self.factor})"

if __name__ == "__main__":
    double = Multiplier(2)
    triple = Multiplier(3)
    print(double(5))
    print(triple(5))
    print(repr(double))
