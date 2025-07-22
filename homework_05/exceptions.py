"""
Объявите следующие исключения:
- LowFuelError
- NotEnoughFuel
- CargoOverload
"""
class LowFuelError(Exception):
    print('Test')

class NotEnoughFuel(Exception):
    pass

class CargoOverload(Exception):
    pass