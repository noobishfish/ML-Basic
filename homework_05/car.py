
from homework_05.base import Vehicle
from homework_05.engine import Engine


class Car(Vehicle):
    engine: Engine

    def set_engine(self, engine: Engine):
        self.engine = engine
