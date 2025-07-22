from homework_05.base import Vehicle
from homework_05 import exceptions

class Plane(Vehicle):
    cargo: int = 0
    max_cargo: int

    def __init__(self, max_cargo, weight = 10, fuel = 10, fuel_consumption = 2):
        super().__init__(weight, fuel, fuel_consumption)
        self.max_cargo = max_cargo

    def load_cargo(self, cargo: int):
        if self.cargo + cargo > self.max_cargo:
            raise exceptions.CargoOverload
        else: self.cargo += cargo

    def remove_all_cargo(self):
        current_cargo = self.cargo
        self.cargo = 0
        return current_cargo

