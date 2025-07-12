"""
Swarm Agent Coordinator - Central hub for agent communication and task distribution
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import redis
from concurrent.futures import ThreadPoolExecutor
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class AgentStatus(Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class Task:
    id: str
    type: str
    priority: int
    payload: Dict[str, Any]
    status: TaskStatus
    created_at: datetime
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self):
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        if self.assigned_at:
            data['assigned_at'] = self.assigned_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data


@dataclass
class Agent:
    id: str
    type: str
    status: AgentStatus
    capabilities: List[str]
    last_heartbeat: datetime
    current_task: Optional[str] = None
    performance_score: float = 1.0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    def to_dict(self):
        data = asdict(self)
        data['status'] = self.status.value
        data['last_heartbeat'] = self.last_heartbeat.isoformat()
        return data


class SwarmCoordinator:
    """Central coordinator for agent swarm management"""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.memory_namespace = "swarm-auto-centralized-1752300927219"
        self.running = False
        
    async def start(self):
        """Start the coordinator"""
        self.running = True
        logger.info("Starting Swarm Coordinator...")
        
        # Start background tasks
        await asyncio.gather(
            self._heartbeat_monitor(),
            self._task_distributor(),
            self._result_aggregator(),
            self._memory_sync()
        )
    
    async def stop(self):
        """Stop the coordinator"""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("Swarm Coordinator stopped")
    
    def register_agent(self, agent_type: str, capabilities: List[str]) -> str:
        """Register a new agent"""
        agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"
        agent = Agent(
            id=agent_id,
            type=agent_type,
            status=AgentStatus.AVAILABLE,
            capabilities=capabilities,
            last_heartbeat=datetime.now()
        )
        self.agents[agent_id] = agent
        
        # Store in Redis for persistence
        self.redis_client.hset(
            f"{self.memory_namespace}:agents",
            agent_id,
            json.dumps(agent.to_dict())
        )
        
        logger.info(f"Registered agent: {agent_id} with capabilities: {capabilities}")
        return agent_id
    
    def heartbeat(self, agent_id: str, status: AgentStatus = AgentStatus.AVAILABLE):
        """Update agent heartbeat"""
        if agent_id in self.agents:
            self.agents[agent_id].last_heartbeat = datetime.now()
            self.agents[agent_id].status = status
            
            # Update Redis
            self.redis_client.hset(
                f"{self.memory_namespace}:agents",
                agent_id,
                json.dumps(self.agents[agent_id].to_dict())
            )
    
    async def submit_task(self, task_type: str, payload: Dict[str, Any], priority: int = 5) -> str:
        """Submit a new task"""
        task_id = f"task_{uuid.uuid4().hex}"
        task = Task(
            id=task_id,
            type=task_type,
            priority=priority,
            payload=payload,
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        self.tasks[task_id] = task
        await self.task_queue.put(task)
        
        # Store in Redis
        self.redis_client.hset(
            f"{self.memory_namespace}:tasks",
            task_id,
            json.dumps(task.to_dict())
        )
        
        logger.info(f"Submitted task: {task_id} of type: {task_type}")
        return task_id
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats"""
        while self.running:
            try:
                current_time = datetime.now()
                for agent_id, agent in list(self.agents.items()):
                    if current_time - agent.last_heartbeat > timedelta(seconds=30):
                        agent.status = AgentStatus.OFFLINE
                        if agent.current_task:
                            # Reassign task if agent went offline
                            task = self.tasks.get(agent.current_task)
                            if task:
                                task.status = TaskStatus.PENDING
                                task.assigned_to = None
                                await self.task_queue.put(task)
                
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
    
    async def _task_distributor(self):
        """Distribute tasks to available agents"""
        while self.running:
            try:
                task = await self.task_queue.get()
                
                # Find suitable agent
                suitable_agents = [
                    agent for agent in self.agents.values()
                    if agent.status == AgentStatus.AVAILABLE 
                    and task.type in agent.capabilities
                ]
                
                if suitable_agents:
                    # Select agent with best performance score
                    best_agent = max(suitable_agents, key=lambda a: a.performance_score)
                    
                    # Assign task
                    task.status = TaskStatus.ASSIGNED
                    task.assigned_to = best_agent.id
                    task.assigned_at = datetime.now()
                    
                    best_agent.status = AgentStatus.BUSY
                    best_agent.current_task = task.id
                    
                    # Notify agent through Redis pub/sub
                    self.redis_client.publish(
                        f"{self.memory_namespace}:agent:{best_agent.id}",
                        json.dumps(task.to_dict())
                    )
                    
                    logger.info(f"Assigned task {task.id} to agent {best_agent.id}")
                else:
                    # No suitable agent available, put task back
                    await asyncio.sleep(1)
                    await self.task_queue.put(task)
                    
            except Exception as e:
                logger.error(f"Task distributor error: {e}")
    
    async def _result_aggregator(self):
        """Aggregate results from agents"""
        while self.running:
            try:
                # Check for completed tasks in Redis
                task_keys = self.redis_client.hkeys(f"{self.memory_namespace}:results")
                
                for task_id in task_keys:
                    result_data = self.redis_client.hget(f"{self.memory_namespace}:results", task_id)
                    if result_data:
                        result = json.loads(result_data)
                        
                        if task_id in self.tasks:
                            task = self.tasks[task_id]
                            task.status = TaskStatus.COMPLETED
                            task.completed_at = datetime.now()
                            task.result = result
                            
                            # Update agent statistics
                            if task.assigned_to in self.agents:
                                agent = self.agents[task.assigned_to]
                                agent.completed_tasks += 1
                                agent.status = AgentStatus.AVAILABLE
                                agent.current_task = None
                                
                                # Update performance score
                                task_duration = (task.completed_at - task.assigned_at).total_seconds()
                                expected_duration = 60  # Expected 60 seconds per task
                                agent.performance_score = (
                                    0.9 * agent.performance_score + 
                                    0.1 * min(2.0, expected_duration / max(task_duration, 1))
                                )
                            
                            # Remove from results
                            self.redis_client.hdel(f"{self.memory_namespace}:results", task_id)
                            
                            logger.info(f"Task {task_id} completed successfully")
                
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Result aggregator error: {e}")
    
    async def _memory_sync(self):
        """Sync memory with other components"""
        while self.running:
            try:
                # Export current state to shared memory
                state = {
                    'timestamp': datetime.now().isoformat(),
                    'agents': {aid: agent.to_dict() for aid, agent in self.agents.items()},
                    'tasks': {tid: task.to_dict() for tid, task in self.tasks.items()},
                    'statistics': self._calculate_statistics()
                }
                
                self.redis_client.set(
                    f"{self.memory_namespace}:coordinator:state",
                    json.dumps(state),
                    ex=300  # Expire after 5 minutes
                )
                
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Memory sync error: {e}")
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate swarm statistics"""
        total_tasks = len(self.tasks)
        completed_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        
        agent_stats = defaultdict(lambda: {'completed': 0, 'failed': 0, 'performance': 0})
        for agent in self.agents.values():
            agent_stats[agent.type]['completed'] += agent.completed_tasks
            agent_stats[agent.type]['failed'] += agent.failed_tasks
            agent_stats[agent.type]['performance'] += agent.performance_score
        
        return {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'success_rate': completed_tasks / total_tasks if total_tasks > 0 else 0,
            'active_agents': sum(1 for a in self.agents.values() if a.status != AgentStatus.OFFLINE),
            'agent_statistics': dict(agent_stats)
        }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        if task_id in self.tasks:
            return self.tasks[task_id].to_dict()
        return None
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent status"""
        if agent_id in self.agents:
            return self.agents[agent_id].to_dict()
        return None


# Example usage
async def main():
    coordinator = SwarmCoordinator()
    
    # Register some agents
    rust_agent_id = coordinator.register_agent("rust_analyzer", ["crypto_analysis", "sequence_parsing"])
    python_agent_id = coordinator.register_agent("python_analyzer", ["visualization", "statistics"])
    
    # Submit some tasks
    task1 = await coordinator.submit_task("crypto_analysis", {"genome": "ecoli_k12", "window_size": 1000})
    task2 = await coordinator.submit_task("visualization", {"data": "frequency_analysis", "type": "heatmap"})
    
    # Start coordinator
    await coordinator.start()


if __name__ == "__main__":
    asyncio.run(main())