import numpy as np


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


class AIbrain_maie:
    TEAM_NAME = "maie"

    def __init__(self):
        self.NAME = f"AIbrain_{self.TEAM_NAME}"
        self.score = 0.0

        self.car_speed = 0.0
        self._steer_memory = 0.0

        self.init_param()
        self.store()

    # -------------------------------------------------
    # Inicializace parametrů
    # -------------------------------------------------
    def init_param(self):
        self.input_dim = None
        self.hidden = 16

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        # === stabilnější mutace ===
        self.mut_sigma_small = 0.02
        self.mut_sigma_big = 0.10
        self.mut_big_prob = 0.02
        self.reset_weight_prob = 0.002

        # === plynulejší řízení ===
        self.steer_smooth = 0.75

    def _init_weights(self, input_dim):
        self.input_dim = input_dim

        lim1 = np.sqrt(6 / (input_dim + self.hidden))
        self.W1 = np.random.uniform(-lim1, lim1, (self.hidden, input_dim))
        self.b1 = np.zeros(self.hidden)

        lim2 = np.sqrt(6 / (self.hidden + 4))
        self.W2 = np.random.uniform(-lim2, lim2, (4, self.hidden))
        self.b2 = np.zeros(4)

    # -------------------------------------------------
    # Rozhodování
    # -------------------------------------------------
    def decide(self, data):
        x = np.array(data, dtype=np.float32)

        if self.W1 is None:
            self._init_weights(len(x))

        # Fix: Ensure x size matches W1 input dimension
        n_in = self.W1.shape[1]
        if x.size < n_in:
            x = np.concatenate([x, np.zeros(n_in - x.size, dtype=np.float32)])
        elif x.size > n_in:
            x = x[:n_in]

        # normalizace paprsků
        mx = max(np.max(x), 1e-6)
        x = 1.0 - np.clip(x / mx, 0, 1)

        # MLP
        h = np.tanh(self.W1 @ x + self.b1)
        out = sigmoid(self.W2 @ h + self.b2)

        # plynulé zatáčení
        steer_raw = out[3] - out[2]
        self._steer_memory = (
            self.steer_smooth * self._steer_memory
            + (1 - self.steer_smooth) * steer_raw
        )

        if self._steer_memory > 0:
            out[3] = 0.5 + abs(self._steer_memory)
            out[2] = 0.5 - abs(self._steer_memory)
        else:
            out[2] = 0.5 + abs(self._steer_memory)
            out[3] = 0.5 - abs(self._steer_memory)

        # lehká preference plynu
        out[0] = np.clip(out[0] + 0.05, 0, 1)

        return out.tolist()

    # -------------------------------------------------
    # Evoluční mutace
    # -------------------------------------------------
    def mutate(self):
        if self.W1 is None:
            return

        big = np.random.rand() < self.mut_big_prob
        sigma = self.mut_sigma_big if big else self.mut_sigma_small

        def mutate_arr(a):
            a = a + np.random.normal(0, sigma, a.shape)
            mask = np.random.rand(*a.shape) < self.reset_weight_prob
            a[mask] = np.random.uniform(-1, 1, np.sum(mask))
            return a

        self.W1 = mutate_arr(self.W1)
        self.b1 = mutate_arr(self.b1)
        self.W2 = mutate_arr(self.W2)
        self.b2 = mutate_arr(self.b2)

        # === KLAMP ===
        np.clip(self.W1, -1.5, 1.5, out=self.W1)
        np.clip(self.W2, -1.5, 1.5, out=self.W2)
        np.clip(self.b1, -1.0, 1.0, out=self.b1)
        np.clip(self.b2, -1.0, 1.0, out=self.b2)

        self._steer_memory = np.clip(self._steer_memory, -1, 1)

        self.store()

    # -------------------------------------------------
    # Skórování
    # -------------------------------------------------
    def calculate_score(self, distance, time, no):
        d = float(distance)
        t = float(time)

        score = d
        score -= 0.02 * t
        score += 0.05 * self.car_speed

        self.score = score

    def passcardata(self, x, y, speed):
        self.car_speed = speed

    def getscore(self):
        return self.score

    # -------------------------------------------------
    # SAVE / LOAD
    # -------------------------------------------------
    def store(self):
        self.parameters = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }

    def get_parameters(self):
        self.store()
        return self.parameters

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.W1 = parameters["W1"]
        self.b1 = parameters["b1"]
        self.W2 = parameters["W2"]
        self.b2 = parameters["b2"]
