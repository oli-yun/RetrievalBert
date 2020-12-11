import numpy as np


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        super().__init__()
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target):
        pred = outputs.data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        self.total += target.size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return float(self.correct) / self.total

    def name(self):
        return 'Accuracy'
