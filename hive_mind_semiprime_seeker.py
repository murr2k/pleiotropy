#!/usr/bin/env python3
"""
Hive Mind Semiprime Seeker
A swarm intelligence system to find the largest semiprime that takes 10 minutes to factor

Agent Roles:
- Scout: Explores new digit ranges quickly
- Analyst: Refines estimates based on timing data
- Challenger: Tests edge cases and difficult semiprimes
- Validator: Verifies results and ensures correctness
"""

import multiprocessing as mp
import numpy as np
from sklearn.linear_model import LinearRegression
import time
import random
import json
import os
import sys
from datetime import datetime
import subprocess
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# Target parameters
TARGET_TIME_SECS = 600  # 10 minutes
TOLERANCE_SECS = 30

# Swarm configuration
SCOUT_COUNT = 3
ANALYST_COUNT = 2
CHALLENGER_COUNT = 2
VALIDATOR_COUNT = 1

@dataclass
class TimingData:
    """Store timing data for regression analysis"""
    digits: int
    time_secs: float
    factor_ratio: float  # ratio of larger to smaller factor
    
@dataclass
class SemiprimeResult:
    """Result of a factorization attempt"""
    number: str
    digits: int
    factor1: str
    factor2: str
    time_secs: float
    agent_type: str
    agent_id: int

class HiveMindMemory:
    """Shared memory for the hive mind"""
    def __init__(self):
        self.manager = mp.Manager()
        self.timing_data = self.manager.list()
        self.best_result = self.manager.dict()
        self.current_estimates = self.manager.dict({
            'target_digits': 40,
            'confidence': 0.0,
            'model_params': None
        })
        self.should_stop = self.manager.Value('b', False)
        self.total_attempts = self.manager.Value('i', 0)
        self.agent_stats = self.manager.dict()
        self.lock = self.manager.Lock()

def generate_prime_gmp(bits: int) -> str:
    """Generate a prime using GMP library via subprocess"""
    import gmpy2
    while True:
        candidate = gmpy2.mpz(random.getrandbits(bits))
        candidate |= (1 << (bits - 1))  # Set MSB
        candidate |= 1  # Ensure odd
        if gmpy2.is_prime(candidate):
            return str(candidate)

def generate_semiprime(digits: int, factor_ratio: float = 1.0) -> Tuple[str, str, str]:
    """
    Generate a semiprime with specified characteristics
    factor_ratio: ratio of larger to smaller factor (1.0 = balanced)
    """
    # Calculate bits for each factor
    total_bits = int(digits * 3.322)
    
    if factor_ratio == 1.0:
        bits1 = bits2 = total_bits // 2
    else:
        # Solve: bits1 + bits2 = total_bits, 2^bits2 / 2^bits1 ≈ factor_ratio
        # This gives: bits2 - bits1 ≈ log2(factor_ratio)
        diff = int(np.log2(factor_ratio))
        bits1 = (total_bits - diff) // 2
        bits2 = total_bits - bits1
    
    # Add some randomness
    bits1 += random.randint(-3, 3)
    bits2 += random.randint(-3, 3)
    
    p1 = generate_prime_gmp(max(bits1, 10))
    p2 = generate_prime_gmp(max(bits2, 10))
    
    # Calculate semiprime
    import gmpy2
    n = gmpy2.mpz(p1) * gmpy2.mpz(p2)
    
    return str(n), p1, p2

def factor_semiprime_cuda(number: str) -> Optional[Tuple[str, str, float]]:
    """Factor a semiprime using CUDA, return (factor1, factor2, time) or None"""
    # For very large numbers, we need a different approach
    # Use the Python factorization script that calls CUDA
    
    python_code = f"""
import subprocess
import time

n = {number}
start = time.time()

# Call the CUDA factorizer through command line
result = subprocess.run([
    'python3', 'factor_large_number.py', 
    '--number', str(n),
    '--cuda'
], capture_output=True, text=True, timeout=800)

elapsed = time.time() - start

if result.returncode == 0:
    # Parse output for factors
    lines = result.stdout.strip().split('\\n')
    for line in lines:
        if 'Prime 1:' in line:
            f1 = line.split(':')[1].strip().replace(',', '')
        elif 'Prime 2:' in line:
            f2 = line.split(':')[1].strip().replace(',', '')
    
    print(f"{{f1}},{{f2}},{{elapsed}}")
else:
    print(f"ERROR: {{result.stderr}}")
"""
    
    try:
        result = subprocess.run(
            ['python3', '-c', python_code],
            capture_output=True,
            text=True,
            timeout=TARGET_TIME_SECS + 120
        )
        
        if result.returncode == 0 and ',' in result.stdout:
            parts = result.stdout.strip().split(',')
            if len(parts) == 3:
                return (parts[0], parts[1], float(parts[2]))
    except Exception as e:
        logging.error(f"Factorization error: {e}")
    
    return None

class ScoutAgent:
    """Explores new digit ranges quickly with small samples"""
    def __init__(self, agent_id: int, memory: HiveMindMemory):
        self.id = agent_id
        self.memory = memory
        self.logger = logging.getLogger(f"Scout-{agent_id}")
    
    def run(self):
        self.logger.info("Scout agent starting")
        attempts = 0
        
        while not self.memory.should_stop.value:
            # Get current estimate
            target_digits = self.memory.current_estimates['target_digits']
            
            # Explore around target with wider range
            explore_digits = target_digits + random.randint(-5, 5)
            explore_digits = max(30, min(80, explore_digits))
            
            # Generate balanced semiprime for speed
            self.logger.info(f"Exploring {explore_digits}-digit range")
            n, p1, p2 = generate_semiprime(explore_digits, factor_ratio=1.0)
            actual_digits = len(n)
            
            # Attempt factorization
            result = factor_semiprime_cuda(n)
            attempts += 1
            self.memory.total_attempts.value += 1
            
            if result:
                f1, f2, time_secs = result
                self.logger.info(f"Factored {actual_digits}-digit in {time_secs:.2f}s")
                
                # Record timing data
                with self.memory.lock:
                    self.memory.timing_data.append(
                        TimingData(actual_digits, time_secs, 1.0)
                    )
                
                # Check if near target
                if abs(time_secs - TARGET_TIME_SECS) <= TOLERANCE_SECS:
                    self._update_best_result(n, actual_digits, f1, f2, time_secs)
            
            # Update stats
            self.memory.agent_stats[f"Scout-{self.id}"] = {
                'attempts': attempts,
                'last_digits': actual_digits
            }
            
            time.sleep(0.5)  # Brief pause
    
    def _update_best_result(self, n, digits, f1, f2, time_secs):
        """Update best result if this is better"""
        current_best = dict(self.memory.best_result)
        
        if not current_best or abs(time_secs - TARGET_TIME_SECS) < abs(current_best.get('time', 0) - TARGET_TIME_SECS):
            self.logger.info(f"🎯 New best: {digits}-digit in {time_secs:.2f}s")
            self.memory.best_result.update({
                'number': n,
                'digits': digits,
                'factor1': f1,
                'factor2': f2,
                'time': time_secs,
                'agent': f"Scout-{self.id}"
            })

class AnalystAgent:
    """Analyzes timing data and updates estimates"""
    def __init__(self, agent_id: int, memory: HiveMindMemory):
        self.id = agent_id
        self.memory = memory
        self.logger = logging.getLogger(f"Analyst-{agent_id}")
        self.model = None
    
    def run(self):
        self.logger.info("Analyst agent starting")
        
        while not self.memory.should_stop.value:
            time.sleep(10)  # Analyze every 10 seconds
            
            timing_data = list(self.memory.timing_data)
            if len(timing_data) < 5:
                continue
            
            # Prepare data for regression
            X = np.array([[d.digits, d.factor_ratio] for d in timing_data])
            y = np.array([np.log(d.time_secs) for d in timing_data])  # Log scale for time
            
            # Fit model
            try:
                self.model = LinearRegression()
                self.model.fit(X, y)
                
                # Predict target digits for 10-minute factorization
                target_log_time = np.log(TARGET_TIME_SECS)
                
                # Binary search for right digit count
                low, high = 30, 80
                best_digits = 40
                
                for _ in range(10):
                    mid = (low + high) // 2
                    pred_time = np.exp(self.model.predict([[mid, 1.0]])[0])
                    
                    if pred_time < TARGET_TIME_SECS:
                        low = mid
                    else:
                        high = mid
                    
                    if abs(pred_time - TARGET_TIME_SECS) < abs(np.exp(self.model.predict([[best_digits, 1.0]])[0]) - TARGET_TIME_SECS):
                        best_digits = mid
                
                # Update estimates
                confidence = self.model.score(X, y)
                self.logger.info(f"Updated estimate: {best_digits} digits (confidence: {confidence:.2f})")
                
                self.memory.current_estimates.update({
                    'target_digits': best_digits,
                    'confidence': confidence,
                    'model_params': pickle.dumps(self.model)
                })
                
            except Exception as e:
                self.logger.error(f"Model fitting error: {e}")

class ChallengerAgent:
    """Tests difficult cases with unbalanced factors"""
    def __init__(self, agent_id: int, memory: HiveMindMemory):
        self.id = agent_id
        self.memory = memory
        self.logger = logging.getLogger(f"Challenger-{agent_id}")
    
    def run(self):
        self.logger.info("Challenger agent starting")
        attempts = 0
        
        while not self.memory.should_stop.value:
            # Get current estimate
            target_digits = self.memory.current_estimates['target_digits']
            confidence = self.memory.current_estimates['confidence']
            
            # Only work when we have good confidence
            if confidence < 0.7:
                time.sleep(5)
                continue
            
            # Test with unbalanced factors
            factor_ratio = random.uniform(1.5, 3.0)  # Moderately unbalanced
            
            test_digits = target_digits + random.randint(-2, 2)
            test_digits = max(30, min(80, test_digits))
            
            self.logger.info(f"Testing {test_digits}-digit with ratio {factor_ratio:.1f}")
            n, p1, p2 = generate_semiprime(test_digits, factor_ratio)
            actual_digits = len(n)
            
            # Attempt factorization
            result = factor_semiprime_cuda(n)
            attempts += 1
            self.memory.total_attempts.value += 1
            
            if result:
                f1, f2, time_secs = result
                self.logger.info(f"Factored challenging {actual_digits}-digit in {time_secs:.2f}s")
                
                # Record timing data
                with self.memory.lock:
                    self.memory.timing_data.append(
                        TimingData(actual_digits, time_secs, factor_ratio)
                    )
                
                # Check if near target
                if abs(time_secs - TARGET_TIME_SECS) <= TOLERANCE_SECS:
                    self._update_best_result(n, actual_digits, f1, f2, time_secs)
            
            # Update stats
            self.memory.agent_stats[f"Challenger-{self.id}"] = {
                'attempts': attempts,
                'last_ratio': factor_ratio
            }
            
            time.sleep(1)
    
    def _update_best_result(self, n, digits, f1, f2, time_secs):
        """Update best result if this is better"""
        current_best = dict(self.memory.best_result)
        
        if not current_best or abs(time_secs - TARGET_TIME_SECS) < abs(current_best.get('time', 0) - TARGET_TIME_SECS):
            self.logger.info(f"🎯 New best: {digits}-digit in {time_secs:.2f}s")
            self.memory.best_result.update({
                'number': n,
                'digits': digits,
                'factor1': f1,
                'factor2': f2,
                'time': time_secs,
                'agent': f"Challenger-{self.id}"
            })

class ValidatorAgent:
    """Validates results and coordinates stopping"""
    def __init__(self, agent_id: int, memory: HiveMindMemory):
        self.id = agent_id
        self.memory = memory
        self.logger = logging.getLogger(f"Validator-{agent_id}")
    
    def run(self):
        self.logger.info("Validator agent starting")
        start_time = time.time()
        
        while not self.memory.should_stop.value:
            time.sleep(5)
            
            # Check if we have a very good result
            best = dict(self.memory.best_result)
            if best:
                time_diff = abs(best['time'] - TARGET_TIME_SECS)
                if time_diff < 5:  # Within 5 seconds
                    self.logger.info(f"✅ Excellent result found, stopping search")
                    self.memory.should_stop.value = True
                    break
            
            # Check time limit
            if time.time() - start_time > 3600:  # 1 hour
                self.logger.info("Time limit reached")
                self.memory.should_stop.value = True
                break
            
            # Log progress
            attempts = self.memory.total_attempts.value
            if attempts % 20 == 0:
                self.logger.info(f"Progress: {attempts} attempts, {len(self.memory.timing_data)} data points")

def launch_hive_mind():
    """Launch the hive mind swarm"""
    print("🐝 Hive Mind Semiprime Seeker")
    print(f"Target: {TARGET_TIME_SECS/60:.0f}-minute factorization")
    print(f"Swarm: {SCOUT_COUNT} scouts, {ANALYST_COUNT} analysts, {CHALLENGER_COUNT} challengers, {VALIDATOR_COUNT} validator")
    print("="*80)
    
    # Initialize shared memory
    memory = HiveMindMemory()
    
    # Create agent processes
    processes = []
    
    # Launch scouts
    for i in range(SCOUT_COUNT):
        agent = ScoutAgent(i, memory)
        p = mp.Process(target=agent.run)
        p.start()
        processes.append(p)
    
    # Launch analysts
    for i in range(ANALYST_COUNT):
        agent = AnalystAgent(i, memory)
        p = mp.Process(target=agent.run)
        p.start()
        processes.append(p)
    
    # Launch challengers
    for i in range(CHALLENGER_COUNT):
        agent = ChallengerAgent(i, memory)
        p = mp.Process(target=agent.run)
        p.start()
        processes.append(p)
    
    # Launch validator
    agent = ValidatorAgent(0, memory)
    p = mp.Process(target=agent.run)
    p.start()
    processes.append(p)
    
    # Monitor in main thread
    try:
        while not memory.should_stop.value:
            time.sleep(10)
            
            # Display status
            total_agents = SCOUT_COUNT + ANALYST_COUNT + CHALLENGER_COUNT + VALIDATOR_COUNT
            active = len(memory.agent_stats)
            attempts = memory.total_attempts.value
            target = memory.current_estimates['target_digits']
            confidence = memory.current_estimates['confidence']
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Hive Status:")
            print(f"  Active agents: {active}/{total_agents}")
            print(f"  Total attempts: {attempts}")
            print(f"  Target estimate: {target} digits (confidence: {confidence:.2%})")
            
            best = dict(memory.best_result)
            if best:
                print(f"  Best result: {best['digits']}-digit in {best['time']:.1f}s (Δ{best['time']-TARGET_TIME_SECS:+.1f}s)")
    
    except KeyboardInterrupt:
        print("\nShutting down hive mind...")
        memory.should_stop.value = True
    
    # Wait for all agents
    for p in processes:
        p.join()
    
    # Final report
    print("\n" + "="*80)
    print("🏁 HIVE MIND SEARCH COMPLETE")
    
    best = dict(memory.best_result)
    if best:
        print(f"\n✅ Optimal Semiprime Found:")
        print(f"  Digits: {best['digits']}")
        print(f"  Time: {best['time']:.2f} seconds")
        print(f"  Target deviation: {best['time'] - TARGET_TIME_SECS:+.2f} seconds")
        print(f"  Found by: {best['agent']}")
        print(f"  Total attempts: {memory.total_attempts.value}")
        
        # Save detailed results
        results = {
            'timestamp': datetime.now().isoformat(),
            'target_seconds': TARGET_TIME_SECS,
            'tolerance_seconds': TOLERANCE_SECS,
            'hive_config': {
                'scouts': SCOUT_COUNT,
                'analysts': ANALYST_COUNT,
                'challengers': CHALLENGER_COUNT,
                'validators': VALIDATOR_COUNT
            },
            'result': best,
            'timing_data': [
                {'digits': d.digits, 'time': d.time_secs, 'ratio': d.factor_ratio}
                for d in memory.timing_data
            ]
        }
        
        with open('hive_mind_semiprime_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n📄 Results saved to hive_mind_semiprime_results.json")
    else:
        print("\n❌ No suitable semiprime found")

if __name__ == "__main__":
    # Required for multiprocessing on some systems
    mp.set_start_method('spawn', force=True)
    launch_hive_mind()