import json
import os
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class ArenaConfig:
    rows: int = 8
    cols: int = 11
    pico_locations: Dict[str, tuple] = None
    server_port: int = 8000
    websocket_port: int = 8765
    bluetooth_scan_interval: float = 1.0
    rssi_timeout: int = 30
    min_rssi: float = -90
    location_update_interval: float = 0.5
    sections: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.pico_locations is None:
            self.pico_locations = {
                'NW': (0, 0),
                'NE': (0, self.cols-1),
                'SW': (self.rows-1, 0),
                'SE': (self.rows-1, self.cols-1)
            }
        if self.sections is None:
            self.sections = {
                'A': [f'A{i}' for i in range(1, self.cols + 1)],
                'B': [f'B{i}' for i in range(1, self.cols + 1)],
                'C': [f'C{i}' for i in range(1, self.cols + 1)],
                'D': [f'D{i}' for i in range(1, self.cols + 1)],
                'E': [f'E{i}' for i in range(1, self.cols + 1)],
                'F': [f'F{i}' for i in range(1, self.cols + 1)],
                'G': [f'G{i}' for i in range(1, self.cols + 1)],
                'H': [f'H{i}' for i in range(1, self.cols + 1)]
            }

class ConfigManager:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.arena_layout = self.generate_arena_layout()
        self.seat_mapping = self.generate_seat_mapping()

    def load_config(self) -> ArenaConfig:
        """Load configuration from file or create default"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config_dict = json.load(f)
                return ArenaConfig(**config_dict)
        return ArenaConfig()

    def save_config(self):
        """Save current configuration to file"""
        with open(self.config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

    def generate_arena_layout(self) -> Dict[str, Dict[str, Any]]:
        """Generate arena layout information"""
        layout = {}
        for row in range(self.config.rows):
            row_letter = chr(65 + row)
            for col in range(1, self.config.cols + 1):
                seat_id = f"{row_letter}{col}"
                layout[seat_id] = {
                    'row': row,
                    'col': col - 1,
                    'section': row_letter,
                    'coordinates': (row, col - 1)
                }
        return layout

    def generate_seat_mapping(self) -> Dict[Tuple[int, int], str]:
        """Generate mapping from coordinates to seat IDs"""
        return {(data['row'], data['col']): seat_id 
                for seat_id, data in self.arena_layout.items()}

    def get_nearest_seat(self, coordinates: Tuple[float, float]) -> str:
        """Find the nearest seat to given coordinates"""
        min_distance = float('inf')
        nearest_seat = None
        
        for seat_id, data in self.arena_layout.items():
            distance = ((coordinates[0] - data['row']) ** 2 + 
                       (coordinates[1] - data['col']) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_seat = seat_id
        
        return nearest_seat

    def get_section_for_seat(self, seat_id: str) -> str:
        """Get the section for a given seat"""
        return self.arena_layout[seat_id]['section']

    def get_section_seats(self, section: str) -> List[str]:
        """Get all seats in a section"""
        return self.config.sections.get(section, [])

    def update_pico_location(self, pico_id: str, location: Tuple[int, int]):
        """Update a Pico device's location"""
        self.config.pico_locations[pico_id] = location
        self.save_config()

    def get_rssi_threshold_for_section(self, section: str) -> float:
        """Get RSSI threshold for a section (for signal strength visualization)"""
        # This could be customized based on arena layout and requirements
        base_threshold = self.config.min_rssi
        section_adjustments = {
            'A': 5,  # Closer to NW/NE Picos
            'B': 3,
            'C': 0,
            'D': 0,
            'E': 0,
            'F': 0,
            'G': 3,
            'H': 5   # Closer to SW/SE Picos
        }
        return base_threshold + section_adjustments.get(section, 0)

    def get_all_sections(self) -> List[str]:
        """Get list of all sections"""
        return list(self.config.sections.keys())

    def get_section_boundaries(self, section: str) -> Dict[str, int]:
        """Get the boundaries of a section"""
        seats = self.config.sections.get(section, [])
        if not seats:
            return {}
            
        rows = [self.arena_layout[seat]['row'] for seat in seats]
        cols = [self.arena_layout[seat]['col'] for seat in seats]
        
        return {
            'min_row': min(rows),
            'max_row': max(rows),
            'min_col': min(cols),
            'max_col': max(cols)
        }

    def export_config(self) -> Dict[str, Any]:
        """Export configuration for frontend use"""
        return {
            'arena': {
                'rows': self.config.rows,
                'cols': self.config.cols,
                'layout': self.arena_layout,
                'sections': self.config.sections
            },
            'picos': self.config.pico_locations,
            'thresholds': {
                'rssi': self.config.min_rssi,
                'timeout': self.config.rssi_timeout
            },
            'update_intervals': {
                'bluetooth': self.config.bluetooth_scan_interval,
                'location': self.config.location_update_interval
            }
        }