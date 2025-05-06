import numpy as np
from scipy.spatial.distance import euclidean
from .constants import *

class TennisEnvironment:
    def __init__(self):
        self.reset()

        # Physics calculations
        self.area = np.pi * (BALL_RADIUS ** 2)
        self.rho_air = AIR_DENSITY
        self.c_d = DRAG_COEFFICIENT

    def reset(self):
        self.ball_position = np.array([COURT_DIMENSIONS[0], COURT_DIMENSIONS[1]/2])
        self.ball_velocity = np.array([0.0, 0.0])
        self.distance_since_last_hit = 0.0
        self.direction = 0  # 0 for left, 1 for right
        self.time = 0.0
        self.done = False
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.ball_position[0],
            self.ball_position[1],
            self.ball_velocity[0],
            self.ball_velocity[1],
            self.distance_since_last_hit,
            self.direction
        ], dtype=np.float32)

    def step(self, action):
        """Execute one time step with physics simulation"""
        if self.done:
            return self._get_state(), 0, True, {}

        # Apply player action (shot impact)
        self.ball_velocity += action

        # Physics simulation
        drag = -0.5 * self.rho_air * self.c_d * self.area * np.linalg.norm(self.ball_velocity) * self.ball_velocity
        acceleration = np.array([0.0, -GRAVITY]) + drag/BALL_MASS

        # Update state
        self.ball_velocity += acceleration * DT
        self.ball_position += self.ball_velocity * DT
        self.time += DT

        # Check boundaries
        if (self.ball_position[0] < 0 or self.ball_position[0] > COURT_DIMENSIONS[0] or
            self.ball_position[1] < 0):
            self.done = True

        reward = self._calculate_reward()
        return self._get_state(), reward, self.done, {}

    def _calculate_reward(self):
        """Reward based on shot quality"""
        target_area = np.array([COURT_DIMENSIONS[0]/2, COURT_DIMENSIONS[1]])
        distance = euclidean(self.ball_position, target_area)
        return np.exp(-distance/5.0)