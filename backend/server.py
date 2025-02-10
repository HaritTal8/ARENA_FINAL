import asyncio
import json
from datetime import datetime
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor

class ArenaLocationSystem:
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
        
        # Physical parameters
        self.reference_power = -65  # Reference power at 1m distance
        self.path_loss_exponent = 2.5  # Initial path loss exponent
        self.grid_size = 1.0  # Size of each grid unit in meters
        
    def calculate_distance(self, rssi):
        """Calculate distance with multipath effects and shadow fading."""
        # Direct path loss calculation
        distance = 10 ** ((self.reference_power - rssi) / (10 * self.path_loss_exponent))
        
        # Add multipath effects
        n_reflections = 3
        total_power = 10**(rssi/10)  # Convert RSSI to power
        
        for i in range(n_reflections):
            reflection_dist = distance * (1 + 0.3 * (i + 1))
            reflection_coeff = 0.4 / (i + 1)  # Decreasing reflection coefficient
            reflection_power = 10**((self.reference_power - 10 * self.path_loss_exponent * 
                                   np.log10(reflection_dist))/10) * reflection_coeff
            total_power += reflection_power
        
        # Include shadow fading
        shadow_std_dev = 2  # dB
        shadow_fading = np.random.normal(0, shadow_std_dev)
        
        return distance * (1 + shadow_fading/10)

    def get_path_loss_exponent(self, rssi_values):
        """Dynamically adjust path loss exponent based on signal patterns."""
        avg_rssi = np.mean(list(rssi_values.values()))
        if avg_rssi > -70:
            return 2.0  # Less obstruction
        elif avg_rssi > -85:
            return 2.5  # Normal indoor
        else:
            return 3.0  # Heavy obstruction

    def calculate_weights(self, rssi_values):
        """Calculate weights based on signal quality and consistency."""
        weights = {}
        rssi_std = np.std(list(rssi_values.values()))
        min_rssi = min(rssi_values.values())
        max_rssi = max(rssi_values.values())
        rssi_range = max_rssi - min_rssi if max_rssi != min_rssi else 1
        
        for sensor, rssi in rssi_values.items():
            # Base weight from signal strength
            weight = 1.0 / (abs(rssi) ** 2)
            
            # Adjust for signal stability
            stability = 1.0 - (rssi_std / 100)
            weight *= max(0.1, stability)
            
            # Consider relative signal strength
            relative_strength = (rssi - min_rssi) / rssi_range
            weight *= (0.5 + 0.5 * relative_strength)
            
            weights[sensor] = weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        for sensor in weights:
            weights[sensor] /= total_weight
            
        return weights

    def calculate_jacobian(self, position):
        """Calculate Jacobian matrix for position refinement."""
        J = np.zeros((len(self.servers), 2))
        for i, (_, server_pos) in enumerate(self.servers.items()):
            server_pos = np.array(server_pos)
            diff = position - server_pos
            dist = np.linalg.norm(diff)
            if dist > 0:
                J[i] = diff / dist
        return J

    def calculate_residuals(self, position, distances):
        """Calculate residuals for position refinement."""
        r = np.zeros(len(self.servers))
        for i, (sensor, server_pos) in enumerate(self.servers.items()):
            server_pos = np.array(server_pos)
            calculated_dist = np.linalg.norm(position - server_pos)
            measured_dist = distances[sensor]
            r[i] = calculated_dist - measured_dist
        return r

    def calculate_position(self, distances, weights):
        """Enhanced position calculation using iterative refinement."""
        # Initial position estimate using weighted centroid
        position = np.zeros(2)
        for sensor, weight in weights.items():
            position += np.array(self.servers[sensor]) * weight
            
        # Convert to numpy array for calculations
        position = np.array(position)
        
        # Iterative refinement using Gauss-Newton
        for _ in range(5):  # Maximum 5 iterations
            J = self.calculate_jacobian(position)
            r = self.calculate_residuals(position, distances)
            
            try:
                delta = np.linalg.pinv(J.T @ J) @ J.T @ r
                position -= delta
                
                if np.linalg.norm(delta) < 0.01:  # Convergence check
                    break
            except np.linalg.LinAlgError:
                break  # Stop if matrix is singular
                
        return position

    def calculate_position_score(self, row, col, rssi_values):
        """Calculate score for a potential position."""
        pos = np.array([row, col])
        score = 0
        
        for sensor, rssi in rssi_values.items():
            server_pos = np.array(self.servers[sensor])
            distance = np.linalg.norm(pos - server_pos)
            expected_rssi = self.reference_power - 10 * self.path_loss_exponent * np.log10(max(1, distance))
            score -= abs(rssi - expected_rssi)
            
        return score

    def refine_location(self, row, col, rssi_values):
        """Refine location prediction using grid constraints."""
        # Check surrounding seats
        surrounding_scores = {}
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                test_row = row + dx
                test_col = col + dy
                if 0 <= test_row < self.rows and 0 <= test_col < self.cols:
                    score = self.calculate_position_score(test_row, test_col, rssi_values)
                    surrounding_scores[(test_row, test_col)] = score
        
        # Find best position
        best_pos = max(surrounding_scores.items(), key=lambda x: x[1])[0]
        return best_pos

    def predict_location(self, rssi_values):
        """Enhanced location prediction with multiple refinement steps."""
        try:
            # Check signal quality
            avg_rssi = np.mean(list(rssi_values.values()))
            rssi_std = np.std(list(rssi_values.values()))
            
            if avg_rssi < -90 or rssi_std > 20:  # Poor signal quality
                return None
            
            # Dynamic path loss adjustment
            self.path_loss_exponent = self.get_path_loss_exponent(rssi_values)
            
            # Calculate distances with enhanced path loss model
            distances = {
                sensor: self.calculate_distance(rssi)
                for sensor, rssi in rssi_values.items()
            }
            
            # Calculate enhanced weights
            weights = self.calculate_weights(rssi_values)
            
            # Get initial position
            position = self.calculate_position(distances, weights)
            
            # Convert to grid coordinates
            row = int(round(position[0]))
            col = int(round(position[1]))
            
            # Refine position
            row, col = self.refine_location(row, col, rssi_values)
            
            # Ensure boundaries
            row = max(0, min(row, self.rows - 1))
            col = max(0, min(col, self.cols - 1))
            
            return f"{chr(65 + row)}{col + 1}"
            
        except Exception as e:
            print(f"Error in location prediction: {e}")
            return None



class BluetoothDevice:
    def __init__(self, address: str, name: str, rssi: dict):
        self.address = address
        self.name = name or "Unknown"
        self.rssi_values = rssi
        self.last_seen = datetime.now()
        self.device_type = self.determine_device_type()
        self.manufacturer = self.determine_manufacturer()

    def determine_device_type(self):
        avg_rssi = sum(self.rssi_values.values()) / len(self.rssi_values)
        
        # Common device name patterns
        name_lower = self.name.lower() if self.name else ""
        if any(x in name_lower for x in ['iphone', 'galaxy', 'pixel']):
            return "Phone"
        elif any(x in name_lower for x in ['macbook', 'laptop', 'pc']):
            return "Laptop"
        elif any(x in name_lower for x in ['airpods', 'buds', 'headphone']):
            return "Audio Device"
        elif any(x in name_lower for x in ['watch', 'band']):
            return "Wearable"
        
        # Fallback to signal strength based classification
        if avg_rssi > -70:
            return "Nearby Device"
        elif avg_rssi > -85:
            return "Mid-Range Device"
        return "Far Device"

    def determine_manufacturer(self):
        # Common manufacturer MAC prefixes
        mac_prefixes = {
            '00:0C:E7': 'Nokia',
            '00:23:76': 'Samsung',
            '00:25:00': 'Apple',
            '18:B4:30': 'Nest',
            '00:1A:11': 'Google',
            '00:1D:A5': 'Microsoft',
            '00:24:9B': 'Huawei',
            'E4:E0:C5': 'Samsung',
            '48:D7:05': 'Apple',
            '34:AB:37': 'Apple',
        }
        
        # Check for manufacturer based on MAC prefix
        for prefix, manufacturer in mac_prefixes.items():
            if self.address.upper().startswith(prefix):
                return manufacturer
                
        return "Unknown"

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
arena_system = ArenaLocationSystem()
devices: Dict[str, BluetoothDevice] = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected")
    
    try:
        while True:
            devices_data = []
            for device in devices.values():
                # Predict location using the arena system
                predicted_seat = arena_system.predict_location(device.rssi_values)
                
                device_data = {
                    "id": device.address,
                    "name": device.name,
                    "rssi_values": device.rssi_values,
                    "predicted_seat": predicted_seat if predicted_seat else "Unknown",  # Handle low confidence predictions
                    "device_type": device.device_type,
                    "manufacturer": device.manufacturer,
                    "timestamp": datetime.now().isoformat()
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
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("WebSocket client disconnected")

async def scan_bluetooth():
    """Scan for actual Bluetooth devices in the area."""
    from bleak import BleakScanner
    
    scanner = BleakScanner()
    
    while True:
        try:
            # Scan for actual devices
            discovered_devices = await scanner.discover()
            
            # Process each discovered device
            for device in discovered_devices:
                # Get RSSI values from each sensor
                # In a real setup, you would get these from your actual sensors
                base_rssi = device.rssi if device.rssi is not None else -100
                
                # Create device data with actual readings
                device_data = {
                    "address": device.address,
                    "name": device.name or "Unknown Device",
                    "rssi_values": {
                        # For now, simulate multiple sensor readings based on the actual RSSI
                        # In production, you would get these from your actual sensors
                        "NW": base_rssi + np.random.normal(0, 2),
                        "NE": base_rssi + np.random.normal(0, 2),
                        "SW": base_rssi + np.random.normal(0, 2),
                        "SE": base_rssi + np.random.normal(0, 2)
                    }
                }

                # Create or update device in our tracking dictionary
                if device_data["address"] not in devices:
                    devices[device_data["address"]] = BluetoothDevice(
                        device_data["address"],
                        device_data["name"],
                        device_data["rssi_values"]
                    )
                else:
                    devices[device_data["address"]].rssi_values = device_data["rssi_values"]
                    
            # Clean up old devices that haven't been seen recently
            current_time = datetime.now()
            devices_to_remove = []
            for addr, device in devices.items():
                if not hasattr(device, 'last_seen'):
                    device.last_seen = current_time
                elif (current_time - device.last_seen).total_seconds() > 30:
                    devices_to_remove.append(addr)
            
            for addr in devices_to_remove:
                del devices[addr]

        except Exception as e:
            print(f"Bluetooth scanning error: {e}")

        await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(scan_bluetooth())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
