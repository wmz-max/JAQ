import math


class CosineTemperature():
    def __init__(self, eta_max=5, eta_min=0.5, total_epoch=120):
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.T_max = total_epoch - 1
        self.update_tau(0)

    def update_tau(self, epoch):
        self.tau = self.eta_min + .5 * (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * (epoch / self.T_max)))

    @staticmethod
    def tau(self):
        return self.tau

