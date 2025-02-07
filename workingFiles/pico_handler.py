import asyncio
import websockets
import json
from typing import Dict, Set
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PicoDevice:
    id: str
    location: str
    last_seen: datetime
    connected: bool

class PicoHandler:
    def __init__(self):
        self.picos: Dict[str, PicoDevice] = {}
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()

    async def register_pico(self, websocket: websockets.WebSocketServerProtocol, pico_id: str):
        """Register a new Pico device"""
        self.picos[pico_id] = PicoDevice(
            id=pico_id,
            location="Unknown",
            last_seen=datetime.now(),
            connected=True
        )
        self.connected_clients.add(websocket)
        print(f"Pico {pico_id} connected")

    async def unregister_pico(self, websocket: websockets.WebSocketServerProtocol, pico_id: str):
        """Unregister a Pico device"""
        if pico_id in self.picos:
            self.picos[pico_id].connected = False
        self.connected_clients.remove(websocket)
        print(f"Pico {pico_id} disconnected")

    async def handle_pico_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """Handle messages from Pico devices"""
        try:
            data = json.loads(message)
            pico_id = data.get('pico_id')
            if not pico_id:
                return
            
            if pico_id not in self.picos:
                await self.register_pico(websocket, pico_id)
            
            # Update last seen timestamp
            self.picos[pico_id].last_seen = datetime.now()
            
            # Process RSSI data
            if 'rssi_data' in data:
                # Forward to main server
                await self.broadcast_rssi_data(pico_id, data['rssi_data'])
                
        except json.JSONDecodeError:
            print(f"Invalid JSON received: {message}")
        except Exception as e:
            print(f"Error processing message: {e}")

    async def broadcast_rssi_data(self, pico_id: str, rssi_data: Dict[str, float]):
        """Broadcast RSSI data to all connected clients"""
        message = json.dumps({
            'type': 'rssi_update',
            'pico_id': pico_id,
            'data': rssi_data,
            'timestamp': datetime.now().isoformat()
        })
        
        disconnected = set()
        for client in self.connected_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected

    async def cleanup_inactive_picos(self):
        """Remove inactive Picos"""
        while True:
            current_time = datetime.now()
            inactive_picos = []
            
            for pico_id, pico in self.picos.items():
                if (current_time - pico.last_seen).seconds > 30:
                    inactive_picos.append(pico_id)
            
            for pico_id in inactive_picos:
                del self.picos[pico_id]
                print(f"Pico {pico_id} removed due to inactivity")
            
            await asyncio.sleep(10)

    async def handle_websocket(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle WebSocket connections"""
        try:
            async for message in websocket:
                await self.handle_pico_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            # Find and unregister the disconnected Pico
            for pico_id, pico in self.picos.items():
                if websocket in self.connected_clients:
                    await self.unregister_pico(websocket, pico_id)
                    break

    async def start_server(self, host: str = "0.0.0.0", port: int = 8765):
        """Start the WebSocket server"""
        cleanup_task = asyncio.create_task(self.cleanup_inactive_picos())
        
        async with websockets.serve(self.handle_websocket, host, port):
            print(f"Pico WebSocket server running on ws://{host}:{port}")
            await asyncio.Future()  # run forever

        cleanup_task.cancel()

if __name__ == "__main__":
    handler = PicoHandler()
    asyncio.run(handler.start_server())