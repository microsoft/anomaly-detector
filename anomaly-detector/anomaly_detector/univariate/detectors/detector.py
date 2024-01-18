import abc


class AnomalyDetector(abc.ABC):
    def detect(self, directions, last_value=None):
        pass
