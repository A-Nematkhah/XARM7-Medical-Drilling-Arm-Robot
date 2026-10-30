"""Shared rest posture (rad) for sim and real — not a MuJoCo constant."""
import numpy as np

# Comfortable elbow-up / wrist-down seed for xArm7 + RH56.
REST_Q = np.array([0.0, -0.3, 0.0, 0.9, 0.0, 1.2, 0.0])

# xArm7 software limits (rad). Unlimited-range joints are ±2π.
XARM7_LO = np.array([-6.2832, -2.059, -6.2832, -0.19198, -6.2832, -1.69297, -6.2832])
XARM7_HI = np.array([6.2832, 2.0944, 6.2832, 3.927, 6.2832, 3.14159, 6.2832])
