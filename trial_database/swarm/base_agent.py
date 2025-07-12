"""
Base Agent Class - Foundation for all swarm agents
"""

import asyncio
import json
import logging
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional
import redis
import signal
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseSwarmAgent(ABC):
    """Base class for all swarm agents"""
    
    def __init__(self, agent_type: str, capabilities: List[str], 
                 redis_host='localhost', redis_port=6379):
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.memory_namespace = "swarm-auto-centralized-1752300927219"
        self.agent_id = None
        self.running = False
        self.current_task = None
        self.pubsub = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Agent {self.agent_id} received shutdown signal")
        self.stop()
        sys.exit(0)
    
    async def start(self):
        """Start the agent"""
        self.running = True
        
        # Register with coordinator
        self._register()
        
        # Subscribe to task channel
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe(f"{self.memory_namespace}:agent:{self.agent_id}")
        
        logger.info(f"Agent {self.agent_id} started")
        
        # Start main loops
        await asyncio.gather(
            self._heartbeat_loop(),
            self._task_listener(),
            self._memory_reporter()
        )
    
    def stop(self):
        """Stop the agent"""
        self.running = False
        if self.pubsub:
            self.pubsub.unsubscribe()
            self.pubsub.close()
        logger.info(f"Agent {self.agent_id} stopped")
    
    def _register(self):
        """Register with the coordinator"""
        # Generate unique agent ID
        import uuid
        self.agent_id = f"{self.agent_type}_{uuid.uuid4().hex[:8]}"
        
        # Register in Redis
        agent_data = {
            'id': self.agent_id,
            'type': self.agent_type,
            'capabilities': self.capabilities,
            'status': 'available',
            'last_heartbeat': datetime.now().isoformat()
        }
        
        self.redis_client.hset(
            f"{self.memory_namespace}:agents",
            self.agent_id,
            json.dumps(agent_data)
        )
        
        logger.info(f"Registered as {self.agent_id}")
    
    async def _heartbeat_loop(self):
        """Send regular heartbeats"""
        while self.running:
            try:
                status = 'busy' if self.current_task else 'available'
                heartbeat_data = {
                    'agent_id': self.agent_id,
                    'status': status,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.redis_client.publish(
                    f"{self.memory_namespace}:heartbeats",
                    json.dumps(heartbeat_data)
                )
                
                # Update agent record
                agent_data = self.redis_client.hget(f"{self.memory_namespace}:agents", self.agent_id)
                if agent_data:
                    agent = json.loads(agent_data)
                    agent['last_heartbeat'] = datetime.now().isoformat()
                    agent['status'] = status
                    self.redis_client.hset(
                        f"{self.memory_namespace}:agents",
                        self.agent_id,
                        json.dumps(agent)
                    )
                
                await asyncio.sleep(10)  # Heartbeat every 10 seconds
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def _task_listener(self):
        """Listen for assigned tasks"""
        while self.running:
            try:
                message = self.pubsub.get_message(timeout=1)
                if message and message['type'] == 'message':
                    task_data = json.loads(message['data'])
                    await self._process_task(task_data)
                
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Task listener error: {e}")
    
    async def _process_task(self, task_data: Dict[str, Any]):
        """Process an assigned task"""
        self.current_task = task_data
        task_id = task_data['id']
        
        logger.info(f"Processing task {task_id}")
        
        try:
            # Update task status
            task_data['status'] = 'in_progress'
            self.redis_client.hset(
                f"{self.memory_namespace}:tasks",
                task_id,
                json.dumps(task_data)
            )
            
            # Execute task
            result = await self.execute_task(task_data)
            
            # Store result
            self.redis_client.hset(
                f"{self.memory_namespace}:results",
                task_id,
                json.dumps(result)
            )
            
            logger.info(f"Task {task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            error_result = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Store error result
            self.redis_client.hset(
                f"{self.memory_namespace}:errors",
                task_id,
                json.dumps(error_result)
            )
        
        finally:
            self.current_task = None
    
    async def _memory_reporter(self):
        """Report agent state to shared memory"""
        while self.running:
            try:
                # Collect agent metrics
                metrics = await self.collect_metrics()
                
                # Store in memory
                self.redis_client.hset(
                    f"{self.memory_namespace}:agent_metrics",
                    self.agent_id,
                    json.dumps({
                        'timestamp': datetime.now().isoformat(),
                        'metrics': metrics
                    })
                )
                
                await asyncio.sleep(30)  # Report every 30 seconds
            except Exception as e:
                logger.error(f"Memory reporter error: {e}")
    
    @abstractmethod
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect agent-specific metrics - must be implemented by subclasses"""
        pass
    
    def save_to_memory(self, key: str, data: Any):
        """Save data to shared memory"""
        full_key = f"{self.memory_namespace}:{self.agent_type}:{key}"
        self.redis_client.set(full_key, json.dumps(data), ex=3600)  # 1 hour expiry
    
    def load_from_memory(self, key: str) -> Optional[Any]:
        """Load data from shared memory"""
        full_key = f"{self.memory_namespace}:{self.agent_type}:{key}"
        data = self.redis_client.get(full_key)
        return json.loads(data) if data else None
    
    def publish_event(self, event_type: str, data: Dict[str, Any]):
        """Publish an event to other agents"""
        event = {
            'type': event_type,
            'source': self.agent_id,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        self.redis_client.publish(
            f"{self.memory_namespace}:events",
            json.dumps(event)
        )
    
    async def wait_for_event(self, event_type: str, timeout: int = 60) -> Optional[Dict[str, Any]]:
        """Wait for a specific event type"""
        event_pubsub = self.redis_client.pubsub()
        event_pubsub.subscribe(f"{self.memory_namespace}:events")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            while asyncio.get_event_loop().time() - start_time < timeout:
                message = event_pubsub.get_message(timeout=1)
                if message and message['type'] == 'message':
                    event = json.loads(message['data'])
                    if event['type'] == event_type:
                        return event
                
                await asyncio.sleep(0.1)
        finally:
            event_pubsub.unsubscribe()
            event_pubsub.close()
        
        return None