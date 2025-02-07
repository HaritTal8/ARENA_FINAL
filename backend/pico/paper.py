#paper.py

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class EnhancedArenaLocationSystem:
    def __init__(self, rows=8, cols=11):
        self.rows = rows
        self.cols = cols
        self.row_labels = [chr(65 + i) for i in range(rows)]
        self.servers = {
            'NW': (0, 0),
            'NE': (0, cols-1),
            'SW': (rows-1, 0),
            'SE': (rows-1, cols-1)
        }
        self.reference_power = -65
        self.frequency = 2.4e9
        self.wavelength = 3e8 / self.frequency
        self.phase_noise_std = 0.1
        self.path_loss_exponent = 2.5

        # ML components
        self.models = {}
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.imputer = SimpleImputer(strategy='mean')

    def calculate_differential_phase(self, pos, server1_pos, server2_pos):
        """Calculate differential phase between two servers."""
        dist1 = np.sqrt(np.sum((np.array(pos) - np.array(server1_pos))**2))
        dist2 = np.sqrt(np.sum((np.array(pos) - np.array(server2_pos))**2))
        
        phase_diff = 2 * np.pi * ((dist1 - dist2) % self.wavelength) / self.wavelength
        # Add phase noise
        phase_diff += np.random.normal(0, self.phase_noise_std)
        return phase_diff

    def calculate_path_loss(self, distance):
        """Enhanced path loss calculation with multipath effects."""
        if distance == 0:
            return self.reference_power
        
        # Direct path loss
        direct_loss = self.reference_power - (10 * self.path_loss_exponent * np.log10(distance))
        
        # Add multipath effects
        n_reflections = 3
        total_power = 10**(direct_loss/10)
        
        for i in range(n_reflections):
            reflection_dist = distance * (1 + 0.3 * (i + 1))
            reflection_loss = self.reference_power - (10 * self.path_loss_exponent * np.log10(reflection_dist))
            reflection_coeff = 0.4 / (i + 1)  # Decreasing reflection coefficient
            total_power += 10**(reflection_loss/10) * reflection_coeff
        
        # Add shadow fading
        shadow_std_dev = 2  # dB
        shadow_fading = np.random.normal(0, shadow_std_dev)
        
        return 10 * np.log10(total_power) + shadow_fading

    def load_real_data(self):
        """Load the real RSSI and phase difference data."""
        measured_values = {
            # Row A
            'A1': {'RSSI_NW': -40.0, 'RSSI_NE': -61.8, 'RSSI_SW': -58.0, 'RSSI_SE': -62.2,
                  'Phase_NW_NE': 0.09, 'Phase_NW_SW': 0.11, 'Phase_NW_SE': 2.12,
                  'Phase_NE_SW': -0.07, 'Phase_NE_SE': 2.08, 'Phase_SW_SE': 2.23},
            'A2': {'RSSI_NW': -53.4, 'RSSI_NE': -65.9, 'RSSI_SW': -52.8, 'RSSI_SE': -63.5,
                  'Phase_NW_NE': 2.38, 'Phase_NW_SW': 2.06, 'Phase_NW_SE': 3.08,
                  'Phase_NE_SW': 5.84, 'Phase_NE_SE': 0.60, 'Phase_SW_SE': 1.08},
            'A3': {'RSSI_NW': -52.8, 'RSSI_NE': -62.1, 'RSSI_SW': -53.5, 'RSSI_SE': -64.5,
                  'Phase_NW_NE': 3.03, 'Phase_NW_SW': 0.39, 'Phase_NW_SE': 1.88,
                  'Phase_NE_SW': 3.68, 'Phase_NE_SE': 4.98, 'Phase_SW_SE': 1.33},
            'A4': {'RSSI_NW': -53.8, 'RSSI_NE': -62.1, 'RSSI_SW': -57.8, 'RSSI_SE': -61.4,
                  'Phase_NW_NE': 0.08, 'Phase_NW_SW': 5.97, 'Phase_NW_SE': 2.82,
                  'Phase_NE_SW': 5.85, 'Phase_NE_SE': 2.67, 'Phase_SW_SE': 3.05},
            'A5': {'RSSI_NW': -55.2, 'RSSI_NE': -58.5, 'RSSI_SW': -57.8, 'RSSI_SE': -61.3,
                  'Phase_NW_NE': 2.20, 'Phase_NW_SW': 4.71, 'Phase_NW_SE': 1.89,
                  'Phase_NE_SW': 2.58, 'Phase_NE_SE': 6.16, 'Phase_SW_SE': 3.65},
            'A6': {'RSSI_NW': -58.0, 'RSSI_NE': -56.1, 'RSSI_SW': -56.9, 'RSSI_SE': -58.8,
                  'Phase_NW_NE': -0.12, 'Phase_NW_SW': 2.64, 'Phase_NW_SE': 2.81,
                  'Phase_NE_SW': 2.57, 'Phase_NE_SE': 2.69, 'Phase_SW_SE': -0.12},
            'A7': {'RSSI_NW': -55.9, 'RSSI_NE': -56.7, 'RSSI_SW': -60.8, 'RSSI_SE': -57.0,
                  'Phase_NW_NE': 4.15, 'Phase_NW_SW': 5.98, 'Phase_NW_SE': 2.68,
                  'Phase_NE_SW': 1.86, 'Phase_NE_SE': 4.71, 'Phase_SW_SE': 2.77},
            'A8': {'RSSI_NW': -64.5, 'RSSI_NE': -51.5, 'RSSI_SW': -59.6, 'RSSI_SE': -53.4,
                  'Phase_NW_NE': 6.19, 'Phase_NW_SW': 2.47, 'Phase_NW_SE': 5.81,
                  'Phase_NE_SW': 2.80, 'Phase_NE_SE': 5.83, 'Phase_SW_SE': 3.12},
            'A9': {'RSSI_NW': -62.0, 'RSSI_NE': -51.7, 'RSSI_SW': -61.9, 'RSSI_SE': -54.2,
                  'Phase_NW_NE': 3.09, 'Phase_NW_SW': 5.04, 'Phase_NW_SE': 3.80,
                  'Phase_NE_SW': 1.66, 'Phase_NE_SE': 0.51, 'Phase_SW_SE': 5.02},
            'A10': {'RSSI_NW': -63.3, 'RSSI_NE': -51.0, 'RSSI_SW': -64.8, 'RSSI_SE': -54.3,
                   'Phase_NW_NE': 3.89, 'Phase_NW_SW': 0.56, 'Phase_NW_SE': 5.65,
                   'Phase_NE_SW': 3.26, 'Phase_NE_SE': 1.75, 'Phase_SW_SE': 5.19},
            'A11': {'RSSI_NW': -62.6, 'RSSI_NE': -48.4, 'RSSI_SW': -62.7, 'RSSI_SE': -53.1,
                   'Phase_NW_NE': 3.28, 'Phase_NW_SW': 2.22, 'Phase_NW_SE': 3.19,
                   'Phase_NE_SW': 5.43, 'Phase_NE_SE': -0.00, 'Phase_SW_SE': 0.84},
            
            # Row B
            'B1': {'RSSI_NW': -37.0, 'RSSI_NE': -62.3, 'RSSI_SW': -56.6, 'RSSI_SE': -65.5,
                  'Phase_NW_NE': 3.94, 'Phase_NW_SW': -0.06, 'Phase_NW_SE': 4.48,
                  'Phase_NE_SW': 2.50, 'Phase_NE_SE': 0.77, 'Phase_SW_SE': 4.48},
            'B2': {'RSSI_NW': -56.4, 'RSSI_NE': -62.1, 'RSSI_SW': -50.7, 'RSSI_SE': -66.3,
                  'Phase_NW_NE': 2.55, 'Phase_NW_SW': 1.86, 'Phase_NW_SE': 3.26,
                  'Phase_NE_SW': 5.64, 'Phase_NE_SE': 0.60, 'Phase_SW_SE': 1.35},
            'B3': {'RSSI_NW': -53.6, 'RSSI_NE': -65.1, 'RSSI_SW': -55.5, 'RSSI_SE': -61.1,
                  'Phase_NW_NE': 3.12, 'Phase_NW_SW': 0.54, 'Phase_NW_SE': 2.06,
                  'Phase_NE_SW': 3.59, 'Phase_NE_SE': 4.96, 'Phase_SW_SE': 1.21},
            'B4': {'RSSI_NW': -52.8, 'RSSI_NE': -60.8, 'RSSI_SW': -56.7, 'RSSI_SE': -61.7,
                  'Phase_NW_NE': 0.03, 'Phase_NW_SW': 5.80, 'Phase_NW_SE': 2.67,
                  'Phase_NE_SW': 5.67, 'Phase_NE_SE': 2.57, 'Phase_SW_SE': 2.98},
            'B5': {'RSSI_NW': -56.8, 'RSSI_NE': -60.8, 'RSSI_SW': -57.8, 'RSSI_SE': -59.2,
                  'Phase_NW_NE': 2.18, 'Phase_NW_SW': 4.88, 'Phase_NW_SE': 1.91,
                  'Phase_NE_SW': 2.50, 'Phase_NE_SE': 6.15, 'Phase_SW_SE': 3.34},
            'B6': {'RSSI_NW': -59.6, 'RSSI_NE': -57.6, 'RSSI_SW': -60.0, 'RSSI_SE': -53.3,
                  'Phase_NW_NE': -0.13, 'Phase_NW_SW': 2.65, 'Phase_NW_SE': 2.64,
                  'Phase_NE_SW': 2.67, 'Phase_NE_SE': 2.66, 'Phase_SW_SE': 0.11},
            'B7': {'RSSI_NW': -60.3, 'RSSI_NE': -56.2, 'RSSI_SW': -57.2, 'RSSI_SE': -58.1,
                  'Phase_NW_NE': 4.17, 'Phase_NW_SW': 6.10, 'Phase_NW_SE': 2.68,
                  'Phase_NE_SW': 1.98, 'Phase_NE_SE': 4.73, 'Phase_SW_SE': 2.82},
            'B8': {'RSSI_NW': -60.6, 'RSSI_NE': -52.2, 'RSSI_SW': -64.2, 'RSSI_SE': -56.4,
                  'Phase_NW_NE': 6.13, 'Phase_NW_SW': 2.55, 'Phase_NW_SE': 5.84,
                  'Phase_NE_SW': 2.72, 'Phase_NE_SE': 6.01, 'Phase_SW_SE': 2.96},
            'B9': {'RSSI_NW': -63.8, 'RSSI_NE': -51.1, 'RSSI_SW': -62.3, 'RSSI_SE': -55.1,
                  'Phase_NW_NE': 3.04, 'Phase_NW_SW': 5.02, 'Phase_NW_SE': 3.64,
                  'Phase_NE_SW': 1.84, 'Phase_NE_SE': 0.33, 'Phase_SW_SE': 4.86},
            'B10': {'RSSI_NW': -65.8, 'RSSI_NE': -54.2, 'RSSI_SW': -63.4, 'RSSI_SE': -51.1,
                   'Phase_NW_NE': 3.67, 'Phase_NW_SW': 0.77, 'Phase_NW_SE': 5.66,
                   'Phase_NE_SW': 3.27, 'Phase_NE_SE': 1.98, 'Phase_SW_SE': 4.85},
            'B11': {'RSSI_NW': -67.2, 'RSSI_NE': -51.1, 'RSSI_SW': -62.2, 'RSSI_SE': -57.0,
                   'Phase_NW_NE': 3.26, 'Phase_NW_SW': 2.10, 'Phase_NW_SE': 3.15,
                   'Phase_NE_SW': 5.38, 'Phase_NE_SE': -0.08, 'Phase_SW_SE': 1.00},
            
            # Row C
            'C1': {'RSSI_NW': -47.1, 'RSSI_NE': -64.8, 'RSSI_SW': -56.8, 'RSSI_SE': -63.7,
                  'Phase_NW_NE': 2.49, 'Phase_NW_SW': -0.09, 'Phase_NW_SE': 3.37,
                  'Phase_NE_SW': 3.84, 'Phase_NE_SE': 0.87, 'Phase_SW_SE': 3.39},
            'C2': {'RSSI_NW': -51.7, 'RSSI_NE': -64.5, 'RSSI_SW': -55.5, 'RSSI_SE': -60.4,
                  'Phase_NW_NE': 2.36, 'Phase_NW_SW': 2.02, 'Phase_NW_SE': 3.24,
                  'Phase_NE_SW': 5.80, 'Phase_NE_SE': 0.78, 'Phase_SW_SE': 1.15},
            'C3': {'RSSI_NW': -50.5, 'RSSI_NE': -57.5, 'RSSI_SW': -55.0, 'RSSI_SE': -64.0,
                  'Phase_NW_NE': 2.90, 'Phase_NW_SW': 0.32, 'Phase_NW_SE': 1.90,
                  'Phase_NE_SW': 3.51, 'Phase_NE_SE': 5.21, 'Phase_SW_SE': 1.16},
            'C4': {'RSSI_NW': -55.8, 'RSSI_NE': -59.2, 'RSSI_SW': -57.8, 'RSSI_SE': -59.5,
                  'Phase_NW_NE': 0.15, 'Phase_NW_SW': 5.88, 'Phase_NW_SE': 2.82,
                  'Phase_NE_SW': 5.78, 'Phase_NE_SE': 2.76, 'Phase_SW_SE': 3.19},
            'C5': {'RSSI_NW': -54.6, 'RSSI_NE': -59.1, 'RSSI_SW': -59.4, 'RSSI_SE': -57.3,
                  'Phase_NW_NE': 2.18, 'Phase_NW_SW': 4.70, 'Phase_NW_SE': 1.95,
                  'Phase_NE_SW': 2.69, 'Phase_NE_SE': 5.85, 'Phase_SW_SE': 3.60},
            'C6': {'RSSI_NW': -54.8, 'RSSI_NE': -59.2, 'RSSI_SW': -55.2, 'RSSI_SE': -62.0,
                  'Phase_NW_NE': 0.01, 'Phase_NW_SW': 2.52, 'Phase_NW_SE': 2.80,
                  'Phase_NE_SW': 2.66, 'Phase_NE_SE': 2.56, 'Phase_SW_SE': -0.12},
            'C7': {'RSSI_NW': -58.7, 'RSSI_NE': -59.3, 'RSSI_SW': -60.5, 'RSSI_SE': -55.9,
                  'Phase_NW_NE': 4.16, 'Phase_NW_SW': 6.12, 'Phase_NW_SE': 2.61,
                  'Phase_NE_SW': 1.51, 'Phase_NE_SE': 4.78, 'Phase_SW_SE': 3.01},
            'C8': {'RSSI_NW': -61.4, 'RSSI_NE': -51.6, 'RSSI_SW': -61.8, 'RSSI_SE': -57.3,
                  'Phase_NW_NE': 6.16, 'Phase_NW_SW': 2.53, 'Phase_NW_SE': 5.84,
                  'Phase_NE_SW': 2.92, 'Phase_NE_SE': 5.93, 'Phase_SW_SE': 3.02},
            'C9': {'RSSI_NW': -63.0, 'RSSI_NE': -50.9, 'RSSI_SW': -58.1, 'RSSI_SE': -55.6,
                  'Phase_NW_NE': 3.18, 'Phase_NW_SW': 5.19, 'Phase_NW_SE': 3.55,
                  'Phase_NE_SW': 1.85, 'Phase_NE_SE': 0.44, 'Phase_SW_SE': 4.81},
            'C10': {'RSSI_NW': -62.8, 'RSSI_NE': -52.1, 'RSSI_SW': -63.7, 'RSSI_SE': -55.9,
                   'Phase_NW_NE': 3.82, 'Phase_NW_SW': 0.81, 'Phase_NW_SE': 5.68,
                   'Phase_NE_SW': 3.18, 'Phase_NE_SE': 2.07, 'Phase_SW_SE': 5.16},
            'C11': {'RSSI_NW': -62.8, 'RSSI_NE': -50.7, 'RSSI_SW': -65.8, 'RSSI_SE': -53.8,
                   'Phase_NW_NE': 3.20, 'Phase_NW_SW': 2.14, 'Phase_NW_SE': 3.39,
                   'Phase_NE_SW': 5.19, 'Phase_NE_SE': 0.01, 'Phase_SW_SE': 1.07},
            
            # Row D
            'D1': {'RSSI_NW': -55.6, 'RSSI_NE': -62.2, 'RSSI_SW': -54.7, 'RSSI_SE': -65.3,
                  'Phase_NW_NE': 2.90, 'Phase_NW_SW': 0.04, 'Phase_NW_SE': 5.24,
                  'Phase_NE_SW': 3.28, 'Phase_NE_SE': 2.28, 'Phase_SW_SE': 5.12},
            'D2': {'RSSI_NW': -53.5, 'RSSI_NE': -64.7, 'RSSI_SW': -53.8, 'RSSI_SE': -64.1,
                  'Phase_NW_NE': 2.31, 'Phase_NW_SW': 2.02, 'Phase_NW_SE': 3.19,
                  'Phase_NE_SW': 5.79, 'Phase_NE_SE': 0.71, 'Phase_SW_SE': 1.16},
            'D3': {'RSSI_NW': -52.5, 'RSSI_NE': -61.2, 'RSSI_SW': -54.3, 'RSSI_SE': -63.5,
                  'Phase_NW_NE': 3.03, 'Phase_NW_SW': 0.27, 'Phase_NW_SE': 1.89,
                  'Phase_NE_SW': 3.52, 'Phase_NE_SE': 5.04, 'Phase_SW_SE': 1.39},
            'D4': {'RSSI_NW': -55.8, 'RSSI_NE': -60.7, 'RSSI_SW': -54.4, 'RSSI_SE': -61.7,
                  'Phase_NW_NE': -0.08, 'Phase_NW_SW': 6.04, 'Phase_NW_SE': 2.84,
                  'Phase_NE_SW': 5.80, 'Phase_NE_SE': 2.63, 'Phase_SW_SE': 2.89},
            'D5': {'RSSI_NW': -56.0, 'RSSI_NE': -62.1, 'RSSI_SW': -54.7, 'RSSI_SE': -55.0,
                  'Phase_NW_NE': 1.98, 'Phase_NW_SW': 4.69, 'Phase_NW_SE': 1.99,
                  'Phase_NE_SW': 2.60, 'Phase_NE_SE': 6.23, 'Phase_SW_SE': 3.55},
            'D6': {'RSSI_NW': -57.9, 'RSSI_NE': -60.9, 'RSSI_SW': -57.0, 'RSSI_SE': -59.7,
                  'Phase_NW_NE': 0.01, 'Phase_NW_SW': 2.81, 'Phase_NW_SE': 2.74,
                  'Phase_NE_SW': 2.72, 'Phase_NE_SE': 2.62, 'Phase_SW_SE': -0.08},
            'D7': {'RSSI_NW': -58.0, 'RSSI_NE': -57.9, 'RSSI_SW': -57.9, 'RSSI_SE': -57.2,
                  'Phase_NW_NE': 4.08, 'Phase_NW_SW': 6.21, 'Phase_NW_SE': 2.43,
                  'Phase_NE_SW': 2.04, 'Phase_NE_SE': 4.65, 'Phase_SW_SE': 2.65},
            'D8': {'RSSI_NW': -62.4, 'RSSI_NE': -53.1, 'RSSI_SW': -61.9, 'RSSI_SE': -55.7,
                  'Phase_NW_NE': 6.20, 'Phase_NW_SW': 2.63, 'Phase_NW_SE': 5.94,
                  'Phase_NE_SW': 2.95, 'Phase_NE_SE': 5.89, 'Phase_SW_SE': 3.02},
            'D9': {'RSSI_NW': -59.7, 'RSSI_NE': -55.1, 'RSSI_SW': -65.0, 'RSSI_SE': -61.2,
                  'Phase_NW_NE': 3.12, 'Phase_NW_SW': 5.03, 'Phase_NW_SE': 3.61,
                  'Phase_NE_SW': 1.83, 'Phase_NE_SE': 0.50, 'Phase_SW_SE': 4.80},
            'D10': {'RSSI_NW': -65.8, 'RSSI_NE': -52.8, 'RSSI_SW': -65.2, 'RSSI_SE': -53.9,
                   'Phase_NW_NE': 3.88, 'Phase_NW_SW': 0.69, 'Phase_NW_SE': 5.78,
                   'Phase_NE_SW': 3.20, 'Phase_NE_SE': 1.92, 'Phase_SW_SE': 4.90},
            'D11': {'RSSI_NW': -61.9, 'RSSI_NE': -53.8, 'RSSI_SW': -62.3, 'RSSI_SE': -54.8,
                   'Phase_NW_NE': 3.14, 'Phase_NW_SW': 2.35, 'Phase_NW_SE': 3.34,
                   'Phase_NE_SW': 5.38, 'Phase_NE_SE': 0.04, 'Phase_SW_SE': 1.13},

            # Row E
            'E1': {'RSSI_NW': -56.2, 'RSSI_NE': -68.0, 'RSSI_SW': -48.7, 'RSSI_SE': -61.9,
                  'Phase_NW_NE': 5.37, 'Phase_NW_SW': 0.02, 'Phase_NW_SE': 2.97,
                  'Phase_NE_SW': 0.96, 'Phase_NE_SE': 3.95, 'Phase_SW_SE': 2.91},
            'E2': {'RSSI_NW': -51.3, 'RSSI_NE': -63.5, 'RSSI_SW': -52.6, 'RSSI_SE': -63.4,
                  'Phase_NW_NE': 2.49, 'Phase_NW_SW': 1.89, 'Phase_NW_SE': 3.17,
                  'Phase_NE_SW': 5.71, 'Phase_NE_SE': 0.67, 'Phase_SW_SE': 1.22},
            'E3': {'RSSI_NW': -53.3, 'RSSI_NE': -61.8, 'RSSI_SW': -56.8, 'RSSI_SE': -64.2,
                  'Phase_NW_NE': 3.11, 'Phase_NW_SW': 0.35, 'Phase_NW_SE': 2.13,
                  'Phase_NE_SW': 3.49, 'Phase_NE_SE': 5.10, 'Phase_SW_SE': 1.40},
            'E4': {'RSSI_NW': -57.5, 'RSSI_NE': -58.5, 'RSSI_SW': -58.3, 'RSSI_SE': -61.7,
                  'Phase_NW_NE': 0.03, 'Phase_NW_SW': 6.03, 'Phase_NW_SE': 2.53,
                  'Phase_NE_SW': 5.76, 'Phase_NE_SE': 2.73, 'Phase_SW_SE': 3.06},
            'E5': {'RSSI_NW': -56.0, 'RSSI_NE': -59.3, 'RSSI_SW': -59.0, 'RSSI_SE': -60.6,
                  'Phase_NW_NE': 2.14, 'Phase_NW_SW': 4.62, 'Phase_NW_SE': 1.82,
                  'Phase_NE_SW': 2.52, 'Phase_NE_SE': 6.02, 'Phase_SW_SE': 3.65},
            'E6': {'RSSI_NW': -56.9, 'RSSI_NE': -56.6, 'RSSI_SW': -58.1, 'RSSI_SE': -60.2,
                  'Phase_NW_NE': 0.03, 'Phase_NW_SW': 2.64, 'Phase_NW_SE': 2.64,
                  'Phase_NE_SW': 2.70, 'Phase_NE_SE': 2.67, 'Phase_SW_SE': 0.16},
            'E7': {'RSSI_NW': -62.0, 'RSSI_NE': -54.7, 'RSSI_SW': -61.3, 'RSSI_SE': -58.6,
                  'Phase_NW_NE': 4.18, 'Phase_NW_SW': 6.09, 'Phase_NW_SE': 2.65,
                  'Phase_NE_SW': 1.88, 'Phase_NE_SE': 4.55, 'Phase_SW_SE': 2.75},
            'E8': {'RSSI_NW': -60.1, 'RSSI_NE': -54.5, 'RSSI_SW': -62.1, 'RSSI_SE': -55.7,
                  'Phase_NW_NE': 6.07, 'Phase_NW_SW': 2.77, 'Phase_NW_SE': 5.86,
                  'Phase_NE_SW': 2.89, 'Phase_NE_SE': 6.02, 'Phase_SW_SE': 3.14},
            'E9': {'RSSI_NW': -60.3, 'RSSI_NE': -53.7, 'RSSI_SW': -60.0, 'RSSI_SE': -54.0,
                  'Phase_NW_NE': 3.09, 'Phase_NW_SW': 5.05, 'Phase_NW_SE': 3.64,
                  'Phase_NE_SW': 1.98, 'Phase_NE_SE': 0.30, 'Phase_SW_SE': 4.92},
            'E10': {'RSSI_NW': -66.3, 'RSSI_NE': -51.0, 'RSSI_SW': -64.8, 'RSSI_SE': -54.7,
                   'Phase_NW_NE': 3.65, 'Phase_NW_SW': 0.58, 'Phase_NW_SE': 5.67,
                   'Phase_NE_SW': 3.08, 'Phase_NE_SE': 1.92, 'Phase_SW_SE': 4.98},
            'E11': {'RSSI_NW': -59.6, 'RSSI_NE': -50.2, 'RSSI_SW': -65.5, 'RSSI_SE': -54.6,
                   'Phase_NW_NE': 3.43, 'Phase_NW_SW': 2.27, 'Phase_NW_SE': 3.40,
                   'Phase_NE_SW': 5.24, 'Phase_NE_SE': 0.12, 'Phase_SW_SE': 1.06},

            # Row F
            'F1': {'RSSI_NW': -58.1, 'RSSI_NE': -62.5, 'RSSI_SW': -45.9, 'RSSI_SE': -64.6,
                  'Phase_NW_NE': 3.37, 'Phase_NW_SW': -0.04, 'Phase_NW_SE': 2.41,
                  'Phase_NE_SW': 2.93, 'Phase_NE_SE': 5.37, 'Phase_SW_SE': 2.77},
            'F2': {'RSSI_NW': -54.0, 'RSSI_NE': -62.5, 'RSSI_SW': -53.8, 'RSSI_SE': -64.5,
                  'Phase_NW_NE': 2.52, 'Phase_NW_SW': 2.00, 'Phase_NW_SE': 3.25,
                  'Phase_NE_SW': 5.83, 'Phase_NE_SE': 0.70, 'Phase_SW_SE': 1.17},
            'F3': {'RSSI_NW': -54.0, 'RSSI_NE': -61.9, 'RSSI_SW': -56.8, 'RSSI_SE': -64.2,
                  'Phase_NW_NE': 3.06, 'Phase_NW_SW': 0.44, 'Phase_NW_SE': 1.69,
                  'Phase_NE_SW': 3.56, 'Phase_NE_SE': 4.95, 'Phase_SW_SE': 1.28},
            'F4': {'RSSI_NW': -56.9, 'RSSI_NE': -59.8, 'RSSI_SW': -56.1, 'RSSI_SE': -61.1,
                  'Phase_NW_NE': 0.05, 'Phase_NW_SW': 5.78, 'Phase_NW_SE': 2.72,
                  'Phase_NE_SW': 5.79, 'Phase_NE_SE': 2.55, 'Phase_SW_SE': 3.17},
            'F5': {'RSSI_NW': -57.7, 'RSSI_NE': -59.8, 'RSSI_SW': -57.8, 'RSSI_SE': -58.5,
                  'Phase_NW_NE': 2.10, 'Phase_NW_SW': 4.58, 'Phase_NW_SE': 1.97,
                  'Phase_NE_SW': 2.69, 'Phase_NE_SE': 6.18, 'Phase_SW_SE': 3.66},
            'F6': {'RSSI_NW': -58.0, 'RSSI_NE': -60.1, 'RSSI_SW': -57.4, 'RSSI_SE': -61.1,
                  'Phase_NW_NE': -0.05, 'Phase_NW_SW': 2.64, 'Phase_NW_SE': 2.54,
                  'Phase_NE_SW': 2.65, 'Phase_NE_SE': 2.76, 'Phase_SW_SE': -0.07},
            'F7': {'RSSI_NW': -57.2, 'RSSI_NE': -53.3, 'RSSI_SW': -62.2, 'RSSI_SE': -56.4,
                  'Phase_NW_NE': 4.14, 'Phase_NW_SW': 6.22, 'Phase_NW_SE': 2.71,
                  'Phase_NE_SW': 2.09, 'Phase_NE_SE': 4.67, 'Phase_SW_SE': 2.83},
            'F8': {'RSSI_NW': -62.8, 'RSSI_NE': -52.4, 'RSSI_SW': -62.4, 'RSSI_SE': -56.4,
                  'Phase_NW_NE': 6.25, 'Phase_NW_SW': 2.68, 'Phase_NW_SE': 5.78,
                  'Phase_NE_SW': 2.69, 'Phase_NE_SE': 5.97, 'Phase_SW_SE': 3.14},
            'F9': {'RSSI_NW': -62.3, 'RSSI_NE': -54.6, 'RSSI_SW': -62.2, 'RSSI_SE': -57.0,
                  'Phase_NW_NE': 3.26, 'Phase_NW_SW': 5.00, 'Phase_NW_SE': 3.55,
                  'Phase_NE_SW': 1.73, 'Phase_NE_SE': 0.43, 'Phase_SW_SE': 4.85},
            'F10': {'RSSI_NW': -59.8, 'RSSI_NE': -50.2, 'RSSI_SW': -58.6, 'RSSI_SE': -51.8,
                   'Phase_NW_NE': 3.84, 'Phase_NW_SW': 0.51, 'Phase_NW_SE': 5.75,
                   'Phase_NE_SW': 3.14, 'Phase_NE_SE': 1.97, 'Phase_SW_SE': 4.94},
            'F11': {'RSSI_NW': -64.2, 'RSSI_NE': -51.1, 'RSSI_SW': -63.1, 'RSSI_SE': -56.5,
                   'Phase_NW_NE': 3.36, 'Phase_NW_SW': 2.29, 'Phase_NW_SE': 3.32,
                   'Phase_NE_SW': 5.33, 'Phase_NE_SE': -0.07, 'Phase_SW_SE': 1.18},

                   # Row G
            'G1': {'RSSI_NW': -56.5, 'RSSI_NE': -62.6, 'RSSI_SW': -38.6, 'RSSI_SE': -65.0,
                  'Phase_NW_NE': 4.32, 'Phase_NW_SW': 0.04, 'Phase_NW_SE': 3.86,
                  'Phase_NE_SW': 1.72, 'Phase_NE_SE': 5.81, 'Phase_SW_SE': 3.76},
            'G2': {'RSSI_NW': -47.9, 'RSSI_NE': -68.4, 'RSSI_SW': -56.9, 'RSSI_SE': -63.6,
                  'Phase_NW_NE': 2.39, 'Phase_NW_SW': 1.91, 'Phase_NW_SE': 3.35,
                  'Phase_NE_SW': 5.86, 'Phase_NE_SE': 0.68, 'Phase_SW_SE': 0.98},
            'G3': {'RSSI_NW': -54.1, 'RSSI_NE': -61.2, 'RSSI_SW': -59.7, 'RSSI_SE': -57.7,
                  'Phase_NW_NE': 3.16, 'Phase_NW_SW': 0.43, 'Phase_NW_SE': 1.96,
                  'Phase_NE_SW': 3.60, 'Phase_NE_SE': 5.15, 'Phase_SW_SE': 1.29},
            'G4': {'RSSI_NW': -57.6, 'RSSI_NE': -61.2, 'RSSI_SW': -55.0, 'RSSI_SE': -61.7,
                  'Phase_NW_NE': 0.27, 'Phase_NW_SW': 6.10, 'Phase_NW_SE': 2.76,
                  'Phase_NE_SW': 5.79, 'Phase_NE_SE': 2.55, 'Phase_SW_SE': 3.12},
            'G5': {'RSSI_NW': -56.1, 'RSSI_NE': -64.4, 'RSSI_SW': -57.9, 'RSSI_SE': -60.8,
                  'Phase_NW_NE': 2.15, 'Phase_NW_SW': 4.71, 'Phase_NW_SE': 1.85,
                  'Phase_NE_SW': 2.63, 'Phase_NE_SE': 6.17, 'Phase_SW_SE': 3.60},
            'G6': {'RSSI_NW': -59.3, 'RSSI_NE': -56.8, 'RSSI_SW': -58.8, 'RSSI_SE': -58.8,
                  'Phase_NW_NE': -0.06, 'Phase_NW_SW': 2.63, 'Phase_NW_SE': 2.61,
                  'Phase_NE_SW': 2.63, 'Phase_NE_SE': 2.67, 'Phase_SW_SE': -0.09},
            'G7': {'RSSI_NW': -60.5, 'RSSI_NE': -57.5, 'RSSI_SW': -58.0, 'RSSI_SE': -60.9,
                  'Phase_NW_NE': 4.38, 'Phase_NW_SW': 6.12, 'Phase_NW_SE': 2.51,
                  'Phase_NE_SW': 2.05, 'Phase_NE_SE': 4.77, 'Phase_SW_SE': 2.66},
            'G8': {'RSSI_NW': -61.6, 'RSSI_NE': -54.3, 'RSSI_SW': -66.8, 'RSSI_SE': -54.8,
                  'Phase_NW_NE': 6.33, 'Phase_NW_SW': 2.61, 'Phase_NW_SE': 5.77,
                  'Phase_NE_SW': 2.83, 'Phase_NE_SE': 5.90, 'Phase_SW_SE': 3.11},
            'G9': {'RSSI_NW': -62.0, 'RSSI_NE': -54.8, 'RSSI_SW': -59.2, 'RSSI_SE': -58.0,
                  'Phase_NW_NE': 3.19, 'Phase_NW_SW': 5.04, 'Phase_NW_SE': 3.61,
                  'Phase_NE_SW': 2.03, 'Phase_NE_SE': 0.25, 'Phase_SW_SE': 5.17},
            'G10': {'RSSI_NW': -61.8, 'RSSI_NE': -50.6, 'RSSI_SW': -63.5, 'RSSI_SE': -51.9,
                   'Phase_NW_NE': 3.72, 'Phase_NW_SW': 0.59, 'Phase_NW_SE': 5.64,
                   'Phase_NE_SW': 2.99, 'Phase_NE_SE': 2.05, 'Phase_SW_SE': 5.22},
            'G11': {'RSSI_NW': -66.2, 'RSSI_NE': -52.0, 'RSSI_SW': -65.5, 'RSSI_SE': -56.1,
                   'Phase_NW_NE': 3.20, 'Phase_NW_SW': 2.33, 'Phase_NW_SE': 3.24,
                   'Phase_NE_SW': 5.20, 'Phase_NE_SE': 0.02, 'Phase_SW_SE': 0.88},

            # Row H
            'H1': {'RSSI_NW': -59.4, 'RSSI_NE': -65.5, 'RSSI_SW': -40.0, 'RSSI_SE': -63.4,
                  'Phase_NW_NE': 2.22, 'Phase_NW_SW': -0.09, 'Phase_NW_SE': 0.10,
                  'Phase_NE_SW': 4.10, 'Phase_NE_SE': 4.23, 'Phase_SW_SE': -0.00},
            'H2': {'RSSI_NW': -50.4, 'RSSI_NE': -64.9, 'RSSI_SW': -54.1, 'RSSI_SE': -62.9,
                  'Phase_NW_NE': 2.55, 'Phase_NW_SW': 2.07, 'Phase_NW_SE': 3.26,
                  'Phase_NE_SW': 5.61, 'Phase_NE_SE': 0.81, 'Phase_SW_SE': 1.22},
            'H3': {'RSSI_NW': -55.2, 'RSSI_NE': -62.8, 'RSSI_SW': -54.0, 'RSSI_SE': -59.9,
                  'Phase_NW_NE': 3.31, 'Phase_NW_SW': 0.50, 'Phase_NW_SE': 1.79,
                  'Phase_NE_SW': 3.61, 'Phase_NE_SE': 5.02, 'Phase_SW_SE': 1.42},
            'H4': {'RSSI_NW': -53.8, 'RSSI_NE': -61.3, 'RSSI_SW': -56.6, 'RSSI_SE': -61.7,
                  'Phase_NW_NE': -0.05, 'Phase_NW_SW': 5.87, 'Phase_NW_SE': 2.84,
                  'Phase_NE_SW': 5.66, 'Phase_NE_SE': 2.60, 'Phase_SW_SE': 3.21},
            'H5': {'RSSI_NW': -60.0, 'RSSI_NE': -62.8, 'RSSI_SW': -55.7, 'RSSI_SE': -61.7,
                  'Phase_NW_NE': 2.12, 'Phase_NW_SW': 4.66, 'Phase_NW_SE': 1.93,
                  'Phase_NE_SW': 2.54, 'Phase_NE_SE': 2.74, 'Phase_SW_SE': 0.30},
            'H6': {'RSSI_NW': -59.6, 'RSSI_NE': -56.9, 'RSSI_SW': -62.0, 'RSSI_SE': -59.9,
                  'Phase_NW_NE': -0.07, 'Phase_NW_SW': 2.82, 'Phase_NW_SE': 2.71,
                  'Phase_NE_SW': 2.54, 'Phase_NE_SE': 2.74, 'Phase_SW_SE': 0.30},
            'H7': {'RSSI_NW': -57.6, 'RSSI_NE': -54.9, 'RSSI_SW': -59.9, 'RSSI_SE': -54.0,
                  'Phase_NW_NE': 4.08, 'Phase_NW_SW': 6.09, 'Phase_NW_SE': 2.64,
                  'Phase_NE_SW': 2.26, 'Phase_NE_SE': 4.72, 'Phase_SW_SE': 2.60},
            'H8': {'RSSI_NW': -62.2, 'RSSI_NE': -56.1, 'RSSI_SW': -61.7, 'RSSI_SE': -54.6,
                  'Phase_NW_NE': 6.14, 'Phase_NW_SW': 2.92, 'Phase_NW_SE': 5.84,
                  'Phase_NE_SW': 2.71, 'Phase_NE_SE': 5.86, 'Phase_SW_SE': 3.07},
            'H9': {'RSSI_NW': -60.0, 'RSSI_NE': -56.3, 'RSSI_SW': -65.1, 'RSSI_SE': -54.4,
                  'Phase_NW_NE': 3.25, 'Phase_NW_SW': 5.03, 'Phase_NW_SE': 3.62,
                  'Phase_NE_SW': 1.75, 'Phase_NE_SE': 0.43, 'Phase_SW_SE': 4.97},
            'H10': {'RSSI_NW': -62.3, 'RSSI_NE': -55.0, 'RSSI_SW': -61.6, 'RSSI_SE': -55.1,
                   'Phase_NW_NE': 3.72, 'Phase_NW_SW': 0.49, 'Phase_NW_SE': 5.68,
                   'Phase_NE_SW': 3.10, 'Phase_NE_SE': 1.86, 'Phase_SW_SE': 5.03},
            'H11': {'RSSI_NW': -64.3, 'RSSI_NE': -48.9, 'RSSI_SW': -62.1, 'RSSI_SE': -53.7,
                   'Phase_NW_NE': 3.36, 'Phase_NW_SW': 2.36, 'Phase_NW_SE': 3.26,
                   'Phase_NE_SW': 5.19, 'Phase_NE_SE': 0.14, 'Phase_SW_SE': 1.04}
        }
        
        data = []
        for row_idx, row_label in enumerate(self.row_labels):
            for col in range(self.cols):
                seat_id = f"{row_label}{col+1}"
                if seat_id in measured_values:
                    features = {
                        'x': row_idx,
                        'y': col,
                        **measured_values[seat_id]
                    }
                    data.append(features)
        
        return pd.DataFrame(data)

    def train_enhanced_model(self):
        """Train an enhanced machine learning model with polynomial features."""
        print("\nTraining enhanced model with multiple algorithms...")
        df = self.load_real_data()
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['x', 'y']]
        X = df[feature_cols]
        y = df[['x', 'y']]
        
        # Apply preprocessing
        X_scaled = self.scaler.fit_transform(X)
        X_poly = self.poly.fit_transform(X_scaled)
        
        # Train Gradient Boosting model
        gb_model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=3,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        # Train models
        self.models['gradient_boost'] = MultiOutputRegressor(gb_model)
        self.models['random_forest'] = MultiOutputRegressor(rf_model)
        
        self.models['gradient_boost'].fit(X_poly, y)
        self.models['random_forest'].fit(X_poly, y)
        
        return self.models

    def predict_location(self, measurements):
        """Predict location using ensemble of models."""
        if not self.models:
            raise ValueError("Models not trained. Call train_enhanced_model() first.")
            
        # Prepare features
        features = pd.DataFrame([measurements])
        features_scaled = self.scaler.transform(features)
        features_poly = self.poly.transform(features_scaled)
        
        # Get predictions from all models
        predictions = []
        for model in self.models.values():
            pred = model.predict(features_poly)
            predictions.append(pred[0])
        
        # Ensemble prediction (weighted average)
        weights = [0.6, 0.4]  # Giving more weight to gradient boosting
        final_prediction = np.average(predictions, axis=0, weights=weights)
        return final_prediction

    def evaluate_all_seats(self):
        """Evaluate predictions for all seats."""
        print("\n=== Detailed Seat Analysis ===\n")
        df = self.load_real_data()
        
        results = []
        # Process all seats in order
        for row_idx, row_label in enumerate(self.row_labels):
            for col in range(self.cols):
                seat_id = f"{row_label}{col+1}"
                seat_row = df[
                    (df['x'] == row_idx) & 
                    (df['y'] == col)
                ]
                
                if not seat_row.empty:
                    measurements = {col: seat_row.iloc[0][col] for col in seat_row.columns if col not in ['x', 'y']}
                    pred_loc = self.predict_location(measurements)
                    
                    error = np.sqrt((pred_loc[0] - row_idx)**2 + (pred_loc[1] - col)**2)
                    
                    print(f"Seat {seat_id}:")
                    print(f"Actual position: Row {float(row_idx):.2f}, Column {float(col):.2f}")
                    print(f"Predicted position: Row {pred_loc[0]:.2f}, Column {pred_loc[1]:.2f}")
                    print(f"Prediction error: {error:.2f} grid units\n")
                    
                    results.append({
                        'seat_id': seat_id,
                        'actual_x': row_idx,
                        'actual_y': col,
                        'pred_x': pred_loc[0],
                        'pred_y': pred_loc[1],
                        'error': error
                    })
        
        return pd.DataFrame(results)

    def visualize_arena(self, results_df):
        """Visualize the arena with prediction accuracy."""
        print("\n=== Arena Visualization ===\n")
        print("    " + "   ".join([f"{i+1:2d}" for i in range(self.cols)]))
        
        for i in range(self.rows):
            row_label = self.row_labels[i]
            print(f"{row_label} ", end="")
            
            for j in range(self.cols):
                seat_id = f"{row_label}{j+1}"
                seat_result = results_df[results_df['seat_id'] == seat_id]
                
                if not seat_result.empty:
                    error = seat_result['error'].iloc[0]
                    if error < 0.5:
                        print(" ✓ ", end="")
                    elif error < 1.0:
                        print(" △ ", end="")
                    else:
                        print(" × ", end="")
                else:
                    print(" · ", end="")
            print()
        
        print("\nLegend:")
        print("✓ - Accurate prediction (error < 0.5 grid units)")
        print("△ - Moderate accuracy (error < 1.0 grid units)")
        print("× - Low accuracy (error > 1.0 grid units)")
        print("· - No data point")

if __name__ == "__main__":
    # Initialize and run the system
    print("Initializing Enhanced Arena Location System...")
    arena = EnhancedArenaLocationSystem()
    
    # Train enhanced model
    arena.train_enhanced_model()
    
    # Evaluate all seats
    results = arena.evaluate_all_seats()
    
    # Visualize predictions
    arena.visualize_arena(results)