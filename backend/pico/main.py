#main.py
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures

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
        self.model = None
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        
        # Enhanced model parameters
        self.path_loss_exponent = 2.5
        self.reference_power = -40
        self.frequency = 2.4e9
        self.wavelength = 3e8 / self.frequency
        self.phase_noise_std = 0.1

    def calculate_differential_phase(self, pos, server1_pos, server2_pos):
        """Calculate differential phase between two servers."""
        dist1 = np.sqrt(np.sum((np.array(pos) - np.array(server1_pos))**2))
        dist2 = np.sqrt(np.sum((np.array(pos) - np.array(server2_pos))**2))
        
        phase_diff = 2 * np.pi * ((dist1 - dist2) % self.wavelength) / self.wavelength
        # Add phase noise
        phase_diff += np.random.normal(0, self.phase_noise_std)
        return phase_diff

    def calculate_features(self, pos):
        """Calculate all features for a given position."""
        rssi_values = {}
        phase_diffs = {}
        relative_distances = {}
        
        # Calculate RSSI values
        for server_name, server_pos in self.servers.items():
            distance = np.sqrt(np.sum((np.array(pos) - np.array(server_pos))**2))
            path_loss = self.calculate_path_loss(distance)
            rssi_values[f'RSSI_{server_name}'] = path_loss
            
            # Calculate relative distances
            relative_distances[f'Dist_{server_name}'] = distance
        
        # Calculate differential phase differences
        servers = list(self.servers.keys())
        for i in range(len(servers)):
            for j in range(i + 1, len(servers)):
                phase_diff = self.calculate_differential_phase(
                    pos, 
                    self.servers[servers[i]], 
                    self.servers[servers[j]]
                )
                phase_diffs[f'Phase_{servers[i]}_{servers[j]}'] = phase_diff
        
        # Calculate RSSI ratios and differences
        for i in range(len(servers)):
            for j in range(i + 1, len(servers)):
                rssi_ratio = rssi_values[f'RSSI_{servers[i]}'] / rssi_values[f'RSSI_{servers[j]}']
                rssi_diff = rssi_values[f'RSSI_{servers[i]}'] - rssi_values[f'RSSI_{servers[j]}']
                rssi_values[f'RSSI_ratio_{servers[i]}_{servers[j]}'] = rssi_ratio
                rssi_values[f'RSSI_diff_{servers[i]}_{servers[j]}'] = rssi_diff
        
        return {**rssi_values, **phase_diffs, **relative_distances}

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

    def generate_training_data(self, n_samples=5000):
        """Generate enhanced training data with more samples."""
        data = []
        print("\nGenerating training data with following parameters:")
        print(f"Path Loss Exponent: {self.path_loss_exponent}")
        print(f"Reference Power: {self.reference_power} dBm")
        print(f"Frequency: {self.frequency/1e9} GHz")
        print(f"Number of samples: {n_samples}")
        
        # Generate grid points
        grid_points = []
        for i in range(self.rows):
            for j in range(self.cols):
                grid_points.append((i, j))
        
        # Generate random points
        for _ in range(n_samples - len(grid_points)):
            x = np.random.uniform(0, self.rows-1)
            y = np.random.uniform(0, self.cols-1)
            grid_points.append((x, y))
        
        # Calculate features for all points
        for pos in grid_points:
            features = self.calculate_features(pos)
            data.append({
                'x': pos[0],
                'y': pos[1],
                **features
            })
            
        return pd.DataFrame(data)

    def train_model(self):
        """Train an enhanced machine learning model."""
        print("\nTraining enhanced model...")
        df = self.generate_training_data()
        
        # Separate features and targets
        feature_cols = [col for col in df.columns if col not in ['x', 'y']]
        X = df[feature_cols]
        y = df[['x', 'y']]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Generate polynomial features
        X_poly = self.poly.fit_transform(X_scaled)
        
        # Train gradient boosting model
        base_model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
        
        self.model = MultiOutputRegressor(base_model)
        print("Training model with polynomial features...")
        self.model.fit(X_poly, y)
        
        return self.model

    def predict_location(self, measurements):
        """Predict location using the enhanced model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Scale features
        measurements_scaled = self.scaler.transform(
            pd.DataFrame([measurements])
        )
        
        # Generate polynomial features
        measurements_poly = self.poly.transform(measurements_scaled)
        
        # Predict
        prediction = self.model.predict(measurements_poly)
        return prediction[0]

    def generate_seat_rssi(self):
        """Generate RSSI values for all seats."""
        seat_data = {}
        print("\nCalculating RSSI and phase values for each seat:")
        
        for i, row in enumerate(self.row_labels):
            for j in range(self.cols):
                seat_id = f"{row}{j+1}"
                pos = (i, j)
                
                # Calculate all features
                features = self.calculate_features(pos)
                
                seat_data[seat_id] = {
                    'x': i,
                    'y': j,
                    **features
                }
                
                # Print RSSI values for each seat
                print(f"\nSeat {seat_id}:")
                print("RSSI Values (dBm):", end=' ')
                for server in self.servers:
                    print(f"{server}: {features[f'RSSI_{server}']:.1f}", end=' | ')
                print("\nPhase Differences (rad):", end=' ')
                servers = list(self.servers.keys())
                for i in range(len(servers)):
                    for j in range(i + 1, len(servers)):
                        key = f'Phase_{servers[i]}_{servers[j]}'
                        print(f"{servers[i]}-{servers[j]}: {features[key]:.2f}", end=' | ')
                
        return seat_data

    def visualize_arena(self, seat_data):
        """Visualize the arena with enhanced display."""
        print("\n\n=== Arena Visualization ===\n")
        
        # Print column headers
        print("    ", end="")
        for j in range(self.cols):
            print(f"{j+1:^35}", end="")
        print("\n")
        
        # Print each row
        for i, row in enumerate(self.row_labels):
            print(f"{row}   ", end="")
            
            for j in range(self.cols):
                seat_id = f"{row}{j+1}"
                seat = seat_data[seat_id]
                
                # Get all measurements
                measurements = {k: v for k, v in seat.items() 
                              if k not in ['x', 'y']}
                
                # Predict location
                pred_loc = self.predict_location(measurements)
                error = np.sqrt((pred_loc[0] - seat['x'])**2 + 
                              (pred_loc[1] - seat['y'])**2)
                
                # Determine accuracy indicator
                if error < 0.5:
                    status = "✓"
                elif error < 1.0:
                    status = "△"
                else:
                    status = "×"
                
                # Display RSSI values for NW and NE servers
                print(f"{status}[NW:{seat['RSSI_NW']:>4.1f} NE:{seat['RSSI_NE']:>4.1f}] ", end="")
            print("\n")
        
        print("\nLegend:")
        print("✓ - Accurate prediction (error < 0.5)")
        print("△ - Moderate accuracy (error < 1.0)")
        print("× - Low accuracy")
        print("[NW/NE] - RSSI values from Northwest/Northeast servers (dBm)")

if __name__ == "__main__":
    # Initialize and run the enhanced system
    print("Initializing Enhanced Arena Location System...")
    arena = EnhancedArenaLocationSystem()
    
    # Train model
    arena.train_model()
    
    # Generate and visualize seat data
    seat_data = arena.generate_seat_rssi()
    
    # Visualize
    arena.visualize_arena(seat_data)