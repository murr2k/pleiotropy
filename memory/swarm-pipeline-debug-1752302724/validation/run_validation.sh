#!/bin/bash
# Run comprehensive validation of the pleiotropy detection pipeline

echo "🎭 PIPELINE VALIDATION ORCHESTRATOR"
echo "=================================="
echo "Memory Namespace: swarm-pipeline-debug-1752302724"
echo "Start Time: $(date)"
echo "=================================="

# Set up environment
cd /home/murr2k/projects/agentic/pleiotropy

# Check if real pipeline exists
if [ -f "pleiotropy_core" ] && [ -x "pleiotropy_core" ]; then
    echo "✅ Using real pipeline: pleiotropy_core"
else
    echo "⚠️  Real pipeline not found, will use mock pipeline for testing"
fi

# Generate synthetic data first
echo -e "\n📊 Generating synthetic test data..."
python3 memory/swarm-pipeline-debug-1752302724/validation/synthetic_data_generator.py

# Run validation orchestration
echo -e "\n🚀 Starting validation orchestration..."
python3 memory/swarm-pipeline-debug-1752302724/validation/validation_orchestrator.py

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "\n✅ Validation completed successfully!"
else
    echo -e "\n❌ Validation failed!"
fi

echo -e "\n=================================="
echo "End Time: $(date)"
echo "=================================="