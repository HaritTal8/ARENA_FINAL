import asyncio
import logging
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from arena_server import ArenaServer
from pico_handler import PicoHandler
from config_manager import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArenaSystem:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.arena_server = ArenaServer()
        self.pico_handler = PicoHandler()
        
        # Initialize FastAPI app
        self.app = FastAPI(title="Arena Interactive System")
        self.setup_app()

    def setup_app(self):
        """Setup FastAPI application"""
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Mount static files
        self.app.mount("/static", StaticFiles(directory="static"), name="static")

        # Add routes
        @self.app.get("/api/config")
        async def get_config():
            return self.config_manager.export_config()

        @self.app.get("/api/status")
        async def get_status():
            return {
                "picos": {
                    pico_id: {
                        "connected": pico.connected,
                        "last_seen": pico.last_seen.isoformat(),
                        "location": pico.location
                    }
                    for pico_id, pico in self.pico_handler.picos.items()
                },
                "devices": len(self.arena_server.devices),
                "active_sections": self.get_active_sections()
            }

    def get_active_sections(self) -> Dict[str, bool]:
        """Get active status for each section based on device presence"""
        sections = {section: False for section in self.config_manager.get_all_sections()}
        
        for device in self.arena_server.devices.values():
            if device.seat:
                section = self.config_manager.get_section_for_seat(device.seat)
                sections[section] = True
        
        return sections

    async def start(self):
        """Start all system components"""
        try:
            logger.info("Starting Arena Interactive System...")
            
            # Start Pico WebSocket handler
            pico_task = asyncio.create_task(
                self.pico_handler.start_server(
                    port=self.config_manager.config.websocket_port
                )
            )
            
            # Start main server
            server_task = asyncio.create_task(self.arena_server.run())
            
            # Start FastAPI application
            config = uvicorn.Config(
                self.app,
                host="0.0.0.0",
                port=self.config_manager.config.server_port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            api_task = asyncio.create_task(server.serve())
            
            logger.info("All components started successfully")
            
            # Wait for all tasks
            await asyncio.gather(pico_task, server_task, api_task)
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            raise
        finally:
            logger.info("System shutdown")

    async def shutdown(self):
        """Graceful system shutdown"""
        logger.info("Initiating system shutdown...")
        # Cleanup tasks here
        logger.info("System shutdown complete")

if __name__ == "__main__":
    system = ArenaSystem()
    
    try:
        asyncio.run(system.start())
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        asyncio.run(system.shutdown())