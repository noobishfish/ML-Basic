"""
Доработайте класс `Vehicle`
"""

from abc import ABC

from homework_05 import exceptions


class Vehicle(ABC):
    started = False
    weight = 1000
    fuel = 100
    fuel_consumption = 1

    def __init__(self, weight, fuel, fuel_consumption):
        self.weight = weight
        self.fuel = fuel
        self.fuel_consumption = fuel_consumption

    def start(self):
        if not self.started:
            if self.fuel > 0:
                self.started = True
            else:
                raise exceptions.LowFuelError

    def move(self, distance: int):
        if distance * self.fuel_consumption <= self.fuel:
            self.fuel -= distance * self.fuel_consumption
        else:
            raise exceptions.NotEnoughFuel

test2 = Vehicle(1000,20,2)
print(f'{test2.fuel} fuel consumes {test2.fuel_consumption}')

