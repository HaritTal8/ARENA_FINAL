# server.py
import asyncio
import json
import os
import sys
from datetime import datetime

# Add the parent directory to system path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pico.blue import BluetoothTracker
from pico.paper import EnhancedArenaLocationSystem
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize systems
bluetooth_tracker = BluetoothTracker()
arena_system = EnhancedArenaLocationSystem()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected")
    try:
        while True:
            # Get device data from Bluetooth tracker
            devices_data = []
            for device in bluetooth_tracker.devices.values():
                device_data = {
                    "id": device.address,
                    "name": device.name,
                    "rssi_values": {"scanner": device.rssi} if hasattr(device, 'rssi') else {},
                    "device_type": device.device_type,
                    "manufacturer": device.manufacturer
                }
                devices_data.append(device_data)

            # Send update to client
            await websocket.send_json({
                "type": "update",
                "devices": devices_data,
                "timestamp": datetime.now().isoformat()
            })
            
            await asyncio.sleep(1)  # Update interval
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("WebSocket client disconnected")

async def run_server():
    # Train the arena system model
    print("Training arena system model...")
    arena_system.train_enhanced_model()
    
    # Start Bluetooth scanning
    bluetooth_task = asyncio.create_task(bluetooth_tracker.run())
    
    # Run FastAPI server
    config = uvicorn.Config(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        ws_ping_interval=None  # Disable ping/pong for testing
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error: {e}")

# import asyncio
# import json
# import os
# import sys
# from datetime import datetime
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.multioutput import MultiOutputRegressor

# # Add the parent directory to system path for imports
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

# from pico.blue import BluetoothTracker
# from fastapi import FastAPI, WebSocket
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn

# class EnhancedArenaLocationSystem:
#     def __init__(self, rows=8, cols=11):
#         self.rows = rows
#         self.cols = cols
#         self.row_labels = [chr(65 + i) for i in range(rows)]
#         self.servers = {
#             'NW': (0, 0),
#             'NE': (0, cols-1),
#             'SW': (rows-1, 0),
#             'SE': (rows-1, cols-1)
#         }
#         self.model = None
#         self.scaler = StandardScaler()
#         self.poly = PolynomialFeatures(degree=2, include_bias=False)
        
#         # Enhanced model parameters
#         self.path_loss_exponent = 2.5
#         self.reference_power = -40
#         self.frequency = 2.4e9
#         self.wavelength = 3e8 / self.frequency
#         self.phase_noise_std = 0.1

#     def calculate_features(self, pos, rssi_values):
#         """Calculate features for a given position with provided RSSI values."""
#         features = {}
        
#         # Calculate RSSI values for given position
#         for server_name, server_pos in self.servers.items():
#             distance = np.sqrt(np.sum((np.array(pos) - np.array(server_pos))**2))
#             path_loss = self.calculate_path_loss(distance)
#             features[f'RSSI_{server_name}'] = path_loss
        
#         # Add measured RSSI values
#         features.update(rssi_values)
        
#         return features

#     def calculate_path_loss(self, distance):
#         """Enhanced path loss calculation with multipath effects."""
#         if distance == 0:
#             return self.reference_power
        
#         # Direct path loss
#         direct_loss = self.reference_power - (10 * self.path_loss_exponent * np.log10(distance))
        
#         # Add multipath effects
#         n_reflections = 3
#         total_power = 10**(direct_loss/10)
        
#         for i in range(n_reflections):
#             reflection_dist = distance * (1 + 0.3 * (i + 1))
#             reflection_loss = self.reference_power - (10 * self.path_loss_exponent * np.log10(reflection_dist))
#             reflection_coeff = 0.4 / (i + 1)  # Decreasing reflection coefficient
#             total_power += 10**(reflection_loss/10) * reflection_coeff
        
#         # Add shadow fading
#         shadow_std_dev = 2  # dB
#         shadow_fading = np.random.normal(0, shadow_std_dev)
        
#         return 10 * np.log10(total_power) + shadow_fading

#     def generate_training_data(self, n_samples=5000):
#         """Generate enhanced training data with more samples."""
#         data = []
#         print("\nGenerating training data...")
        
#         # Generate grid points
#         grid_points = []
#         for i in range(self.rows):
#             for j in range(self.cols):
#                 grid_points.append((i, j))
        
#         # Generate random points
#         for _ in range(n_samples - len(grid_points)):
#             x = np.random.uniform(0, self.rows-1)
#             y = np.random.uniform(0, self.cols-1)
#             grid_points.append((x, y))
        
#         # Calculate features for all points
#         for pos in grid_points:
#             # Simulate RSSI values for this position
#             rssi_values = {f'RSSI_{server}': self.calculate_features(pos, {})[f'RSSI_{server}'] 
#                            for server in self.servers}
            
#             features = self.calculate_features(pos, rssi_values)
#             data.append({
#                 'x': pos[0],
#                 'y': pos[1],
#                 **features
#             })
            
#         return pd.DataFrame(data)

#     def train_model(self):
#         """Train an enhanced machine learning model."""
#         print("\nTraining enhanced model...")
#         df = self.generate_training_data()
        
#         # Separate features and targets
#         feature_cols = [col for col in df.columns if col not in ['x', 'y']]
#         X = df[feature_cols]
#         y = df[['x', 'y']]
        
#         # Scale features
#         X_scaled = self.scaler.fit_transform(X)
        
#         # Generate polynomial features
#         X_poly = self.poly.fit_transform(X_scaled)
        
#         # Train gradient boosting model
#         base_model = GradientBoostingRegressor(
#             n_estimators=300,
#             learning_rate=0.1,
#             max_depth=6,
#             min_samples_split=4,
#             min_samples_leaf=2,
#             subsample=0.8,
#             random_state=42
#         )
        
#         self.model = MultiOutputRegressor(base_model)
#         print("Training model with polynomial features...")
#         self.model.fit(X_poly, y)
        
#         return self.model

#     def predict_location(self, measurements):
#         """Predict location using the enhanced model."""
#         if self.model is None:
#             raise ValueError("Model not trained. Call train_model() first.")
        
#         # Ensure measurements are in a DataFrame
#         if not isinstance(measurements, pd.DataFrame):
#             measurements = pd.DataFrame([measurements])
        
#         # Scale features
#         measurements_scaled = self.scaler.transform(measurements)
        
#         # Generate polynomial features
#         measurements_poly = self.poly.transform(measurements_scaled)
        
#         # Predict
#         prediction = self.model.predict(measurements_poly)
#         return prediction[0]

#     def convert_position_to_seat(self, pos):
#         """Convert numeric position to seat label."""
#         row = int(pos[0])
#         col = int(pos[1])
        
#         # Convert to seat label
#         row_label = self.row_labels[row]
#         seat_label = f"{row_label}{col+1}"
        
#         return seat_label

# app = FastAPI()

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize systems
# bluetooth_tracker = BluetoothTracker()
# arena_system = EnhancedArenaLocationSystem()

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     print("WebSocket client connected")
    
#     # Ensure model is trained
#     if arena_system.model is None:
#         arena_system.train_model()
    
#     try:
#         while True:
#             # Get device data from Bluetooth tracker
#             devices_data = []
#             for device in bluetooth_tracker.devices.values():
#                 # Prepare RSSI measurements
#                 rssi_values = {
#                     'RSSI_NW': device.rssi_nw if hasattr(device, 'rssi_nw') else -100,
#                     'RSSI_NE': device.rssi_ne if hasattr(device, 'rssi_ne') else -100,
#                     'RSSI_SW': device.rssi_sw if hasattr(device, 'rssi_sw') else -100,
#                     'RSSI_SE': device.rssi_se if hasattr(device, 'rssi_se') else -100
#                 }
                
#                 # Predict location
#                 try:
#                     predicted_pos = arena_system.predict_location(rssi_values)
#                     seat_label = arena_system.convert_position_to_seat(predicted_pos)
#                 except Exception as e:
#                     print(f"Location prediction error: {e}")
#                     seat_label = "Unknown"
                
#                 device_data = {
#                     "id": device.address,
#                     "name": device.name,
#                     "rssi_values": rssi_values,
#                     "predicted_seat": seat_label,
#                     "device_type": device.device_type,
#                     "manufacturer": device.manufacturer
#                 }
#                 devices_data.append(device_data)

#             # Send update to client
#             await websocket.send_json({
#                 "type": "update",
#                 "devices": devices_data,
#                 "timestamp": datetime.now().isoformat()
#             })
            
#             await asyncio.sleep(1)  # Update interval
#     except Exception as e:
#         print(f"WebSocket error: {e}")
#     finally:
#         print("WebSocket client disconnected")

# async def run_server():
#     # Train the arena system model
#     print("Training arena system model...")
#     arena_system.train_model()
    
#     # Start Bluetooth scanning
#     bluetooth_task = asyncio.create_task(bluetooth_tracker.run())
    
#     # Run FastAPI server
#     config = uvicorn.Config(
#         app, 
#         host="0.0.0.0", 
#         port=8000, 
#         log_level="info",
#         ws_ping_interval=None  # Disable ping/pong for testing
#     )
#     server = uvicorn.Server(config)
#     await server.serve()

# if __name__ == "__main__":
#     try:
#         asyncio.run(run_server())
#     except KeyboardInterrupt:
#         print("\nShutting down server...")
#     except Exception as e:
#         print(f"Error: {e}")