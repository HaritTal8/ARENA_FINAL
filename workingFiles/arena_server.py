from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from bleak import BleakScanner
import uvicorn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd

# Data models
@dataclass
class Device:
    id: str
    name: str
    seat: str
    rssi_values: Dict[str, float]
    last_seen: datetime
    device_type: str
    manufacturer: str

# Add this class before ArenaServer class
class EnhancedArenaLocationSystem:
    def __init__(self):
        self.model = MultiOutputRegressor(GradientBoostingRegressor())
        self.scaler = StandardScaler()
        
    def train_enhanced_model(self):
        # Placeholder for training data
        # You should replace this with your actual training data
        training_data = pd.DataFrame({
            'NW': [-60, -70, -80],
            'NE': [-70, -60, -70],
            'SW': [-80, -70, -60],
            'SE': [-70, -80, -70]
        })
        
        locations = np.array([[0, 0], [5, 5], [10, 10]])  # Example coordinates
        
        self.scaler.fit(training_data)
        scaled_data = self.scaler.transform(training_data)
        self.model.fit(scaled_data, locations)
    
    def predict_location(self, rssi_values):
        data = np.array([[
            rssi_values['NW'],
            rssi_values['NE'],
            rssi_values['SW'],
            rssi_values['SE']
        ]])
        scaled_data = self.scaler.transform(data)
        return self.model.predict(scaled_data)[0]
    
    def get_nearest_seat(self, location):
        # Placeholder - implement your seat mapping logic
        return f"Seat {location[0]:.1f},{location[1]:.1f}"

class ArenaServer:
    def __init__(self):
        self.app = FastAPI()
        self.devices: Dict[str, Device] = {}
        self.location_system = EnhancedArenaLocationSystem()
        self.connected_clients: List[WebSocket] = []
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize routes
        self.setup_routes()
        
        # Train location model
        self.location_system.train_enhanced_model()

    def setup_routes(self):
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.connected_clients.append(websocket)
            try:
                while True:
                    await asyncio.sleep(0.1)  # Prevent CPU overload
                    await self.send_updates(websocket)
            except:
                self.connected_clients.remove(websocket)

        @self.app.get("/devices")
        async def get_devices():
            return {
                "devices": [
                    {
                        "id": dev.id,
                        "name": dev.name,
                        "seat": dev.seat,
                        "rssi_values": dev.rssi_values,
                        "last_seen": dev.last_seen.isoformat(),
                        "device_type": dev.device_type,
                        "manufacturer": dev.manufacturer
                    }
                    for dev in self.devices.values()
                ]
            }

    async def send_updates(self, websocket: WebSocket):
        """Send real-time updates to connected clients"""
        try:
            # Prepare device data
            devices_data = []
            for device in self.devices.values():
                # Calculate predicted location
                if all(sensor in device.rssi_values for sensor in ['NW', 'NE', 'SW', 'SE']):
                    location = self.location_system.predict_location(device.rssi_values)
                    seat = self.location_system.get_nearest_seat(location)
                else:
                    seat = "Unknown"
                    location = None

                devices_data.append({
                    "id": device.id,
                    "name": device.name,
                    "seat": seat,
                    "rssi_values": device.rssi_values,
                    "location": location.tolist() if location is not None else None,
                    "device_type": device.device_type,
                    "manufacturer": device.manufacturer
                })

            # Send update
            await websocket.send_json({
                "type": "update",
                "data": {
                    "devices": devices_data,
                    "timestamp": datetime.now().isoformat()
                }
            })
        except Exception as e:
            print(f"Error sending updates: {e}")

    async def process_pico_data(self, pico_id: str, rssi_data: Dict[str, float]):
        """Process RSSI data from Pico devices"""
        for device_id, rssi in rssi_data.items():
            if device_id not in self.devices:
                self.devices[device_id] = Device(
                    id=device_id,
                    name="Unknown Device",
                    seat="",
                    rssi_values={},
                    last_seen=datetime.now(),
                    device_type="Unknown",
                    manufacturer="Unknown"
                )
            
            # Update RSSI values
            self.devices[device_id].rssi_values[pico_id] = rssi
            self.devices[device_id].last_seen = datetime.now()

    async def start_bluetooth_scanner(self):
        """Start Bluetooth scanning"""
        scanner = BleakScanner()
        
        def callback(device, advertising_data):
            if device.address not in self.devices:
                self.devices[device.address] = Device(
                    id=device.address,
                    name=device.name or "Unknown",
                    seat="",
                    rssi_values={},
                    last_seen=datetime.now(),
                    device_type=self.detect_device_type(device.name),
                    manufacturer=self.detect_manufacturer(device.address)
                )
            
            # Update device info
            self.devices[device.address].last_seen = datetime.now()
            if advertising_data.rssi is not None:
                self.devices[device.address].rssi_values['scanner'] = advertising_data.rssi

        await scanner.start()
        scanner.register_detection_callback(callback)

    def detect_device_type(self, name: Optional[str]) -> str:
        """Detect device type from name"""
        if not name:
            return "Unknown"
        
        name = name.lower()
        if any(keyword in name for keyword in ['phone', 'iphone', 'android']):
            return "Phone"
        elif any(keyword in name for keyword in ['airpods', 'buds']):
            return "Audio Device"
        elif any(keyword in name for keyword in ['watch']):
            return "Wearable"
        return "Unknown"

    def detect_manufacturer(self, mac_address: str) -> str:
        """Detect manufacturer from MAC address"""
        manufacturers = {
            "00:00:0C": "Cisco",
            "00:05:02": "Apple",
            "00:23:76": "Samsung",
            "00:25:00": "Apple",
            "E4:E0:C5": "Samsung",
            "48:D7:05": "Apple",
            "34:AB:37": "Apple"
        }
        
        prefix = mac_address[:8].upper()
        return manufacturers.get(prefix, "Unknown")

    async def cleanup_old_devices(self):
        """Remove devices not seen recently"""
        while True:
            current_time = datetime.now()
            to_remove = []
            
            for device_id, device in self.devices.items():
                if (current_time - device.last_seen).seconds > 30:
                    to_remove.append(device_id)
            
            for device_id in to_remove:
                del self.devices[device_id]
            
            await asyncio.sleep(5)

    async def run(self):
        """Run the server"""
        cleanup_task = asyncio.create_task(self.cleanup_old_devices())
        bluetooth_task = asyncio.create_task(self.start_bluetooth_scanner())
        
        config = uvicorn.Config(self.app, host="0.0.0.0", port=8000)
        server = uvicorn.Server(config)
        
        await server.serve()
        
        cleanup_task.cancel()
        bluetooth_task.cancel()

if __name__ == "__main__":
    server = ArenaServer()
    asyncio.run(server.run())