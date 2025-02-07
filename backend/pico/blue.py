# #BLUE.PY

# import asyncio
# import time
# from datetime import datetime
# from bleak import BleakScanner
# from rich.console import Console
# from rich.table import Table
# from rich.live import Live
# import platform
# import re
# import subprocess
# from typing import Dict, Optional

# class BluetoothDevice:
#     def __init__(self, address: str, name: str, rssi: int):
#         self.address = address
#         self.name = name or "Unknown"
#         self.rssi = rssi
#         self.ip_address = "Scanning..."
#         self.last_seen = time.time()
#         self.device_type = "Unknown"
#         self.manufacturer = "Unknown"

# class BluetoothTracker:
#     def __init__(self):
#         self.console = Console()
#         self.devices: Dict[str, BluetoothDevice] = {}
#         self.is_running = True
        
#         # Common manufacturer prefixes
#         self.manufacturers = {
#             "49:AA": "OnePlus",
#             "1B:C9": "Realtek",
#             "00:16": "Samsung",
#             "E4:E0": "Samsung",
#             "48:D7": "Apple",
#             "34:AB": "Apple",
#             "00:25": "Apple",
#             "A0:B4": "Samsung",
#             "6C:2B": "Microsoft",
#             "58:84": "Intel",
#             "00:1A": "Microsoft"
#         }

#         # Device type patterns
#         self.device_patterns = {
#             r"(?i)(airpods|headphone|buds)": "Audio Device",
#             r"(?i)(mouse|keyboard)": "Input Device",
#             r"(?i)(watch|band)": "Wearable",
#             r"(?i)(phone|iphone|android)": "Phone",
#             r"(?i)(laptop|pc|computer)": "Computer",
#             r"(?i)(tablet|pad)": "Tablet"
#         }

#     def get_manufacturer(self, mac_address: str) -> str:
#         """Identify manufacturer from MAC address prefix"""
#         for prefix, manufacturer in self.manufacturers.items():
#             if mac_address.upper().startswith(prefix):
#                 return manufacturer
#         return "Unknown"

#     def get_device_type(self, name: str, rssi: int, manufacturer: str) -> str:
#         """Determine device type based on name patterns and RSSI"""
#         # First try to identify by name
#         if name and name != "Unknown":
#             for pattern, device_type in self.device_patterns.items():
#                 if re.search(pattern, name):
#                     return f"{manufacturer} {device_type}"

#         # If no name match, estimate by signal strength
#         if rssi > -70:
#             return f"{manufacturer} Nearby Device"
#         elif rssi > -85:
#             return f"{manufacturer} Mid-Range Device"
#         else:
#             return f"{manufacturer} Far Device"

#     async def get_ip_address(self, mac: str) -> str:
#         """Attempt to find IP address for MAC address using system ARP table"""
#         try:
#             if platform.system() == "Windows":
#                 cmd = 'arp -a'
#             else:
#                 cmd = 'arp -n'
                
#             process = await asyncio.create_subprocess_shell(
#                 cmd,
#                 stdout=asyncio.subprocess.PIPE,
#                 stderr=asyncio.subprocess.PIPE
#             )
#             stdout, _ = await process.communicate()
#             output = stdout.decode()
            
#             # Clean MAC address for comparison
#             mac_cleaned = mac.replace(':', '-').lower()
            
#             for line in output.splitlines():
#                 if mac_cleaned in line.lower():
#                     ip_match = re.search(r'\d+\.\d+\.\d+\.\d+', line)
#                     if ip_match:
#                         return ip_match.group(0)
                        
#             # Additional Windows-specific method
#             if platform.system() == "Windows":
#                 process = await asyncio.create_subprocess_shell(
#                     'netsh wlan show all',
#                     stdout=asyncio.subprocess.PIPE,
#                     stderr=asyncio.subprocess.PIPE
#                 )
#                 stdout, _ = await process.communicate()
#                 output = stdout.decode()
                
#                 if mac_cleaned in output.lower():
#                     ip_match = re.search(r'\d+\.\d+\.\d+\.\d+', output)
#                     if ip_match:
#                         return ip_match.group(0)
                        
#         except Exception as e:
#             return f"Error: {str(e)}"
            
#         return "Not Found"

#     async def scan_callback(self, device, advertisement_data):
#         """Callback function for each discovered Bluetooth device"""
#         try:
#             # Update or create device entry
#             if device.address not in self.devices:
#                 self.devices[device.address] = BluetoothDevice(
#                     device.address,
#                     device.name,
#                     advertisement_data.rssi
#                 )
#                 # Get manufacturer and type for new devices
#                 manufacturer = self.get_manufacturer(device.address)
#                 self.devices[device.address].manufacturer = manufacturer
#                 self.devices[device.address].device_type = self.get_device_type(
#                     device.name,
#                     advertisement_data.rssi,
#                     manufacturer
#                 )
#                 # Start IP address lookup for new device
#                 self.devices[device.address].ip_address = await self.get_ip_address(device.address)
#             else:
#                 # Update existing device
#                 self.devices[device.address].rssi = advertisement_data.rssi
#                 self.devices[device.address].last_seen = time.time()
                
#         except Exception as e:
#             self.console.print(f"[red]Error in scan callback: {str(e)}[/red]")

#     def create_table(self) -> Table:
#         """Create and populate the display table"""
#         # Create the table with a descriptive title
#         table = Table(
#             title=f"Bluetooth Device Scanner - {datetime.now().strftime('%H:%M:%S')}",
#             title_style="bold blue"
#         )
        
#         # Add columns with specific styles
#         table.add_column("Device Type", style="cyan")
#         table.add_column("Name", style="blue")
#         table.add_column("MAC Address", style="green")
#         table.add_column("IP Address", style="yellow")
#         table.add_column("Signal Strength", style="red")
        
#         current_time = time.time()
        
#         # Sort devices by signal strength
#         sorted_devices = sorted(
#             self.devices.values(),
#             key=lambda x: x.rssi if x.rssi is not None else -100,
#             reverse=True
#         )
        
#         for device in sorted_devices:
#             # Skip devices not seen in last 30 seconds
#             if current_time - device.last_seen > 30:
#                 continue
                
#             # Create signal strength visualization
#             signal_strength = "█" * (abs(device.rssi) // 10)
#             if device.rssi > -70:
#                 signal_display = f"[green]{device.rssi} dBm {signal_strength}[/green]"
#             elif device.rssi > -85:
#                 signal_display = f"[yellow]{device.rssi} dBm {signal_strength}[/yellow]"
#             else:
#                 signal_display = f"[red]{device.rssi} dBm {signal_strength}[/red]"
                
#             table.add_row(
#                 device.device_type,
#                 device.name,
#                 device.address,
#                 device.ip_address,
#                 signal_display
#             )
            
#         return table

#     async def run(self):
#         """Main function to run the Bluetooth tracker"""
#         try:
#             self.console.clear()
#             self.console.print("[bold blue]Starting Bluetooth Scanner...[/bold blue]")
            
#             scanner = BleakScanner(detection_callback=self.scan_callback)
            
#             # Start scanner
#             await scanner.start()
            
#             # Main display loop
#             with Live(self.create_table(), refresh_per_second=1) as live:
#                 while self.is_running:
#                     live.update(self.create_table())
#                     await asyncio.sleep(1)
                    
#             await scanner.stop()
            
#         except KeyboardInterrupt:
#             self.is_running = False
#             self.console.print("\n[yellow]Shutting down...[/yellow]")
#         except Exception as e:
#             self.console.print(f"\n[red]Error in main loop: {str(e)}[/red]")
#         finally:
#             self.is_running = False
#             self.console.print("[green]Scanner stopped successfully.[/green]")

# if __name__ == "__main__":
#     tracker = BluetoothTracker()
#     try:
#         asyncio.run(tracker.run())
#     except KeyboardInterrupt:
#         pass


import asyncio
import time
from datetime import datetime
from bleak import BleakScanner
from rich.console import Console
from rich.table import Table
from rich.live import Live
import platform
import re
import subprocess
from typing import Dict, Optional

class BluetoothDevice:
    def __init__(self, address: str, name: str, rssi: int):
        self.address = address
        self.name = name or "Unknown"
        self.rssi = rssi
        
        # Multi-directional RSSI values
        self.rssi_nw = -100.0  # Northwest sensor
        self.rssi_ne = -100.0  # Northeast sensor
        self.rssi_sw = -100.0  # Southwest sensor
        self.rssi_se = -100.0  # Southeast sensor
        
        self.ip_address = "Scanning..."
        self.last_seen = time.time()
        self.device_type = "Unknown"
        self.manufacturer = "Unknown"
        
        # Sensor confidence values (to track reliability)
        self.sensor_confidence = {
            'nw': 0,
            'ne': 0,
            'sw': 0,
            'se': 0
        }

class BluetoothTracker:
    def __init__(self, sensor_positions=None):
        self.console = Console()
        self.devices: Dict[str, BluetoothDevice] = {}
        self.is_running = True
        
        # Configurable sensor positions (if not provided, use default)
        self.sensor_positions = sensor_positions or {
            'NW': (-1, -1),   # Northwest corner
            'NE': (-1, 1),    # Northeast corner
            'SW': (1, -1),    # Southwest corner
            'SE': (1, 1)      # Southeast corner
        }
        
        # Common manufacturer prefixes
        self.manufacturers = {
            "49:AA": "OnePlus",
            "1B:C9": "Realtek",
            "00:16": "Samsung",
            "E4:E0": "Samsung",
            "48:D7": "Apple",
            "34:AB": "Apple",
            "00:25": "Apple",
            "A0:B4": "Samsung",
            "6C:2B": "Microsoft",
            "58:84": "Intel",
            "00:1A": "Microsoft"
        }

        # Device type patterns
        self.device_patterns = {
            r"(?i)(airpods|headphone|buds)": "Audio Device",
            r"(?i)(mouse|keyboard)": "Input Device",
            r"(?i)(watch|band)": "Wearable",
            r"(?i)(phone|iphone|android)": "Phone",
            r"(?i)(laptop|pc|computer)": "Computer",
            r"(?i)(tablet|pad)": "Tablet"
        }

    def get_manufacturer(self, mac_address: str) -> str:
        """Identify manufacturer from MAC address prefix"""
        for prefix, manufacturer in self.manufacturers.items():
            if mac_address.upper().startswith(prefix):
                return manufacturer
        return "Unknown"

    def get_device_type(self, name: str, rssi: int, manufacturer: str) -> str:
        """Determine device type based on name patterns and RSSI"""
        # First try to identify by name
        if name and name != "Unknown":
            for pattern, device_type in self.device_patterns.items():
                if re.search(pattern, name):
                    return f"{manufacturer} {device_type}"

        # If no name match, estimate by signal strength
        if rssi > -70:
            return f"{manufacturer} Nearby Device"
        elif rssi > -85:
            return f"{manufacturer} Mid-Range Device"
        else:
            return f"{manufacturer} Far Device"

    async def scan_callback(self, device, advertisement_data):
        """Callback function for each discovered Bluetooth device"""
        try:
            # Simulate multi-directional RSSI 
            # In a real setup, this would come from actual sensor readings
            base_rssi = advertisement_data.rssi
            
            # Create or update device
            if device.address not in self.devices:
                # New device
                new_device = BluetoothDevice(
                    device.address,
                    device.name,
                    base_rssi
                )
                
                # Simulate directional RSSI with some variance
                import random
                new_device.rssi_nw = base_rssi + random.uniform(-3, 3)
                new_device.rssi_ne = base_rssi + random.uniform(-3, 3)
                new_device.rssi_sw = base_rssi + random.uniform(-3, 3)
                new_device.rssi_se = base_rssi + random.uniform(-3, 3)
                
                # Get manufacturer and type
                manufacturer = self.get_manufacturer(device.address)
                new_device.manufacturer = manufacturer
                new_device.device_type = self.get_device_type(
                    device.name,
                    base_rssi,
                    manufacturer
                )
                
                self.devices[device.address] = new_device
            else:
                # Update existing device
                device_obj = self.devices[device.address]
                device_obj.rssi = base_rssi
                
                # Update directional RSSI
                import random
                device_obj.rssi_nw = base_rssi + random.uniform(-3, 3)
                device_obj.rssi_ne = base_rssi + random.uniform(-3, 3)
                device_obj.rssi_sw = base_rssi + random.uniform(-3, 3)
                device_obj.rssi_se = base_rssi + random.uniform(-3, 3)
                
                device_obj.last_seen = time.time()
                
        except Exception as e:
            self.console.print(f"[red]Error in scan callback: {str(e)}[/red]")

    def create_table(self) -> Table:
        """Create and populate the display table with multi-directional RSSI"""
        # Create the table with a descriptive title
        table = Table(
            title=f"Bluetooth Device Scanner - {datetime.now().strftime('%H:%M:%S')}",
            title_style="bold blue"
        )
        
        # Add columns with specific styles
        table.add_column("Device Type", style="cyan")
        table.add_column("Name", style="blue")
        table.add_column("MAC Address", style="green")
        table.add_column("NW RSSI", style="red")
        table.add_column("NE RSSI", style="red")
        table.add_column("SW RSSI", style="red")
        table.add_column("SE RSSI", style="red")
        
        current_time = time.time()
        
        # Sort devices by overall signal strength
        sorted_devices = sorted(
            self.devices.values(),
            key=lambda x: x.rssi if x.rssi is not None else -100,
            reverse=True
        )
        
        for device in sorted_devices:
            # Skip devices not seen in last 30 seconds
            if current_time - device.last_seen > 30:
                continue
                
            table.add_row(
                device.device_type,
                device.name,
                device.address,
                f"{device.rssi_nw:.1f} dBm",
                f"{device.rssi_ne:.1f} dBm",
                f"{device.rssi_sw:.1f} dBm",
                f"{device.rssi_se:.1f} dBm"
            )
            
        return table

    async def run(self):
        """Main function to run the Bluetooth tracker"""
        try:
            self.console.clear()
            self.console.print("[bold blue]Starting Bluetooth Scanner...[/bold blue]")
            
            scanner = BleakScanner(detection_callback=self.scan_callback)
            
            # Start scanner
            await scanner.start()
            
            # Main display loop
            with Live(self.create_table(), refresh_per_second=1) as live:
                while self.is_running:
                    live.update(self.create_table())
                    await asyncio.sleep(1)
                    
            await scanner.stop()
            
        except KeyboardInterrupt:
            self.is_running = False
            self.console.print("\n[yellow]Shutting down...[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]Error in main loop: {str(e)}[/red]")
        finally:
            self.is_running = False
            self.console.print("[green]Scanner stopped successfully.[/green]")

if __name__ == "__main__":
    tracker = BluetoothTracker()
    try:
        asyncio.run(tracker.run())
    except KeyboardInterrupt:
        pass