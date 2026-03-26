class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value

    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9

if __name__ == "__main__":
    t = Temperature(25)
    print(f"{t.celsius}C = {t.fahrenheit}F")
    t.fahrenheit = 100
    print(f"{t.celsius}C = {t.fahrenheit}F")
    try:
        t.celsius = -300
    except ValueError as e:
        print(f"Error: {e}")
