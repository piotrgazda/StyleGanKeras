from tensorflow.keras.constraints import Constraint
from tensorflow.keras import backend as K


class ClipWeights(Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)

