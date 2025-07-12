"""
WebSocket connection manager for real-time updates
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from typing import Dict, Set, List, Optional
import json
import asyncio
from datetime import datetime
import logging
from app.core.auth import get_current_agent
from app.models.schemas import WSMessage

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        # Active connections: {client_id: websocket}
        self.active_connections: Dict[str, WebSocket] = {}
        # Subscription mapping: {trial_id: set(client_ids)}
        self.trial_subscriptions: Dict[int, Set[str]] = {}
        # Client info: {client_id: {"agent_name": str, "connected_at": datetime}}
        self.client_info: Dict[str, Dict] = {}
        # Message queue for reliability
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        # Background task for processing messages
        self.processing_task: Optional[asyncio.Task] = None
    
    async def connect(self, websocket: WebSocket, client_id: str, agent_name: str):
        """Accept and register a new connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_info[client_id] = {
            "agent_name": agent_name,
            "connected_at": datetime.utcnow()
        }
        logger.info(f"WebSocket connected: {client_id} (Agent: {agent_name})")
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection_established",
            "data": {
                "client_id": client_id,
                "message": "Connected to trial tracking WebSocket"
            }
        }, client_id)
    
    def disconnect(self, client_id: str):
        """Remove a connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.client_info[client_id]
            
            # Remove from all subscriptions
            for trial_id, subscribers in self.trial_subscriptions.items():
                subscribers.discard(client_id)
            
            # Clean up empty subscription sets
            self.trial_subscriptions = {
                k: v for k, v in self.trial_subscriptions.items() if v
            }
            
            logger.info(f"WebSocket disconnected: {client_id}")
    
    async def subscribe_to_trial(self, client_id: str, trial_id: int):
        """Subscribe a client to trial updates"""
        if trial_id not in self.trial_subscriptions:
            self.trial_subscriptions[trial_id] = set()
        
        self.trial_subscriptions[trial_id].add(client_id)
        
        await self.send_personal_message({
            "type": "subscription_confirmed",
            "data": {
                "trial_id": trial_id,
                "message": f"Subscribed to trial {trial_id} updates"
            }
        }, client_id)
        
        logger.info(f"Client {client_id} subscribed to trial {trial_id}")
    
    async def unsubscribe_from_trial(self, client_id: str, trial_id: int):
        """Unsubscribe a client from trial updates"""
        if trial_id in self.trial_subscriptions:
            self.trial_subscriptions[trial_id].discard(client_id)
            
            if not self.trial_subscriptions[trial_id]:
                del self.trial_subscriptions[trial_id]
        
        await self.send_personal_message({
            "type": "unsubscription_confirmed",
            "data": {
                "trial_id": trial_id,
                "message": f"Unsubscribed from trial {trial_id} updates"
            }
        }, client_id)
        
        logger.info(f"Client {client_id} unsubscribed from trial {trial_id}")
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Send a message to a specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def send_to_trial_subscribers(self, trial_id: int, message: dict):
        """Send a message to all subscribers of a trial"""
        if trial_id in self.trial_subscriptions:
            disconnected_clients = []
            
            for client_id in self.trial_subscriptions[trial_id]:
                if client_id in self.active_connections:
                    try:
                        await self.active_connections[client_id].send_json(message)
                    except Exception as e:
                        logger.error(f"Error sending to {client_id}: {e}")
                        disconnected_clients.append(client_id)
            
            # Clean up disconnected clients
            for client_id in disconnected_clients:
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict):
        """Send a message to all connected clients"""
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()
        
        # Queue the message for processing
        try:
            await self.message_queue.put(message)
        except asyncio.QueueFull:
            logger.warning("Message queue full, dropping oldest message")
            try:
                self.message_queue.get_nowait()
                await self.message_queue.put(message)
            except:
                logger.error("Failed to queue message")
    
    async def process_message_queue(self):
        """Background task to process queued messages"""
        while True:
            try:
                message = await self.message_queue.get()
                
                # Check if this is a trial-specific update
                if "data" in message and "trial_id" in message.get("data", {}):
                    trial_id = message["data"]["trial_id"]
                    await self.send_to_trial_subscribers(trial_id, message)
                else:
                    # Broadcast to all clients
                    disconnected_clients = []
                    
                    for client_id, websocket in self.active_connections.items():
                        try:
                            await websocket.send_json(message)
                        except Exception as e:
                            logger.error(f"Error broadcasting to {client_id}: {e}")
                            disconnected_clients.append(client_id)
                    
                    # Clean up disconnected clients
                    for client_id in disconnected_clients:
                        self.disconnect(client_id)
                
            except Exception as e:
                logger.error(f"Error processing message queue: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on errors
    
    def get_connection_stats(self) -> dict:
        """Get statistics about current connections"""
        total_connections = len(self.active_connections)
        trial_subscription_counts = {
            trial_id: len(subscribers)
            for trial_id, subscribers in self.trial_subscriptions.items()
        }
        
        agents_connected = {}
        for client_id, info in self.client_info.items():
            agent_name = info["agent_name"]
            if agent_name not in agents_connected:
                agents_connected[agent_name] = 0
            agents_connected[agent_name] += 1
        
        return {
            "total_connections": total_connections,
            "trial_subscriptions": trial_subscription_counts,
            "agents_connected": agents_connected,
            "queue_size": self.message_queue.qsize()
        }


# Create global manager instance
manager = ConnectionManager()

# WebSocket router
websocket_router = APIRouter()


@websocket_router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str = Query(...),
    agent_name: str = Query(...)
):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket, client_id, agent_name)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            
            # Handle different message types
            if data.get("type") == "subscribe":
                trial_id = data.get("trial_id")
                if trial_id:
                    await manager.subscribe_to_trial(client_id, trial_id)
            
            elif data.get("type") == "unsubscribe":
                trial_id = data.get("trial_id")
                if trial_id:
                    await manager.unsubscribe_from_trial(client_id, trial_id)
            
            elif data.get("type") == "ping":
                await manager.send_personal_message({
                    "type": "pong",
                    "data": {"timestamp": datetime.utcnow().isoformat()}
                }, client_id)
            
            else:
                await manager.send_personal_message({
                    "type": "error",
                    "data": {"message": f"Unknown message type: {data.get('type')}"}
                }, client_id)
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)


@websocket_router.get("/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    return manager.get_connection_stats()


# Start background task when module is imported
async def start_background_tasks():
    """Start the message processing background task"""
    if manager.processing_task is None:
        manager.processing_task = asyncio.create_task(manager.process_message_queue())
        logger.info("Started WebSocket message processing task")


# Ensure background task is started
asyncio.create_task(start_background_tasks())