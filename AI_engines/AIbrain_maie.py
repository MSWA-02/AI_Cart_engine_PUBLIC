import numpy as np
import os
from datetime import datetime


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


class AIbrain_maie:
    TEAM_NAME = "maie"

    def __init__(self):
        self.NAME = f"AIbrain_{self.TEAM_NAME}"
        self.score = 0.0
        self.prev_distance = 0.0
        self.car_speed = 0.0
        self.steer_mem = 0.0

        self.input_dim = None
        self.hidden = 16

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        # HYPERPARAMETRY PRO EVOLUCI
        self.mut_sigma_small = 0.08
        self.mut_sigma_big = 0.35
        self.mut_big_prob = 0.08
        self.reset_weight_prob = 0.01
        self.steer_smooth = 0.65

    # -------------------------------------------------

    def _init_weights(self, n):
        self.input_dim = n
        actual_input = n + 1  # Raycasts + Car speed
        self.W1 = np.random.randn(self.hidden, actual_input) * 0.3
        self.b1 = np.zeros(self.hidden)
        self.W2 = np.random.randn(2, self.hidden) * 0.3
        self.b2 = np.array([0.5, 0.0]) # Mírný bias pro plyn vpřed

    # -------------------------------------------------

    def decide(self, data):
        x = np.array(data, dtype=np.float32)

        if self.W1 is None or self.input_dim != len(x):
            self._init_weights(len(x))
            
        # normalizace + blízkost překážky
        mx = max(np.max(x), 1e-6)
        x_norm = 1 - np.clip(x / mx, 0, 1)

        # Rozšíření vstupů o aktuální rychlost
        speed_norm = self.car_speed / 500.0
        inputs = np.append(x_norm, speed_norm)

        h = np.tanh(self.W1 @ inputs + self.b1)
        out = self.W2 @ h + self.b2

        # výstupy
        throttle = sigmoid(out[0])
        steer = np.tanh(out[1])

        # plynulé řízení
        self.steer_mem = (
            self.steer_smooth * self.steer_mem
            + (1 - self.steer_smooth) * steer
        )

        # převod na 4 akce
        up = throttle
        down = 0.0  # brzda vypnutá
        left = max(0.0, -self.steer_mem)
        right = max(0.0, self.steer_mem)

        # nouzová brzda - selektivní na přední paprsky
        # data indexy: 3 (-5°), 4 (0°), 5 (5°) jsou v podstatě předek
        front_danger = max(x_norm[3], x_norm[4], x_norm[5])
        if front_danger > 0.92:
            down = 1.0
            up = 0.0

        return [up, down, left, right]

    # -------------------------------------------------

    def mutate(self):
        def m(a):
            # Dual-level mutation
            if np.random.rand() < self.mut_big_prob:
                sigma = self.mut_sigma_big
            else:
                sigma = self.mut_sigma_small
            
            a += np.random.normal(0, sigma, a.shape)
            
            # Random reset
            mask = np.random.rand(*a.shape) < self.reset_weight_prob
            a[mask] = np.random.uniform(-0.5, 0.5, np.sum(mask))
            
            return np.clip(a, -2, 2)

        self.W1 = m(self.W1)
        self.b1 = m(self.b1)
        self.W2 = m(self.W2)
        self.b2 = m(self.b2)

    # -------------------------------------------------

    def calculate_score(self, distance, time, no):
        d = float(distance)
        t = float(time)

        # FAIL FAST – stojí → konec
        if t > 4 and d < 1.5:
            self.score = -100
            self._log_to_file(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')};{no};{d:.2f};{t:.2f};-100.00")
            return

        # Uložíme si progress pro info, ale fitness postavíme jinak
        self.prev_distance = d

        # Fitness funkce zaměřená na RYCHLOST a pokrok
        # 1. Hlavní složka: ujetá vzdálenost (tiles)
        score = d * 15.0
        
        # 2. Průměrná rychlost (d/t) je klíčová pro rychlé projetí
        if t > 0.5:
            score += (d / t) * 25.0
            
        # 3. Odměna za okamžitou rychlost (motivuje nebrzdit zbytečně)
        # car_speed je v pixelech/s (0-500)
        score += (self.car_speed / 5.0) 

        # 4. Penalizace času (aby se nesnažil jen dojet, ale dojet CO NEJRYCHLEJI)
        score -= t * 3.0

        # checkpoint bonusy - výraznější pro povzbuzení k postupu
        if d > 50: score += 100
        if d > 100: score += 400
        if d > 200: score += 1200

        self.score = score
        self._log_to_file(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')};{no};{d:.2f};{t:.2f};{score:.2f}")

    def _log_to_file(self, data_str):
        log_dir = "UserData/LOGS"
        try:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file = os.path.join(log_dir, f"{self.TEAM_NAME}_learning.csv")
            file_exists = os.path.isfile(log_file)
            with open(log_file, "a") as f:
                if not file_exists:
                    f.write("timestamp;individual_no;distance;time;score\n")
                f.write(data_str + "\n")
        except Exception:
            pass

    def passcardata(self, x, y, speed):
        self.car_speed = speed

    def getscore(self):
        return self.score

    # -------------------------------------------------

    def get_parameters(self):
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "mut_sigma_small": self.mut_sigma_small,
            "mut_sigma_big": self.mut_sigma_big,
            "mut_big_prob": self.mut_big_prob,
            "reset_weight_prob": self.reset_weight_prob,
            "steer_smooth": self.steer_smooth,
            "hidden": self.hidden,
            "input_dim": self.input_dim,
        }

    def set_parameters(self, p):
        self.W1 = p["W1"]
        self.b1 = p["b1"]
        self.W2 = p["W2"]
        self.b2 = p["b2"]
        if "mut_sigma_small" in p: self.mut_sigma_small = float(p["mut_sigma_small"])
        if "mut_sigma_big" in p: self.mut_sigma_big = float(p["mut_sigma_big"])
        if "mut_big_prob" in p: self.mut_big_prob = float(p["mut_big_prob"])
        if "reset_weight_prob" in p: self.reset_weight_prob = float(p["reset_weight_prob"])
        if "steer_smooth" in p: self.steer_smooth = float(p["steer_smooth"])
        if "hidden" in p: self.hidden = int(p["hidden"])
        if "input_dim" in p: self.input_dim = int(p["input_dim"])
