"""Shared rest posture (rad) for sim and real — not a MuJoCo constant."""
import numpy as np

# Comfortable elbow-up / wrist-down seed for xArm7 + RH56.
REST_Q = np.array([0.0, -0.3, 0.0, 0.9, 0.0, 1.2, 0.0])
