import numpy as np


class BestChecker:
    def __init__(self):
        self.val_dice_score = np.NINF
        self.epoch = 0

    def is_best(self, val_dice_score, epoch):
        if self.val_dice_score < val_dice_score:
            self.val_dice_score = val_dice_score
            self.epoch = epoch
            return True
        return False
