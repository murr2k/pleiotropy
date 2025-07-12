# Monitoring and Maintenance Guide

This guide provides comprehensive instructions for monitoring the Genomic Pleiotropy Cryptanalysis system and performing routine maintenance tasks.

## Table of Contents

1. [Monitoring Overview](#monitoring-overview)
2. [Grafana Dashboard](#grafana-dashboard)
3. [Prometheus Metrics](#prometheus-metrics)
4. [Health Checks](#health-checks)
5. [Log Management](#log-management)
6. [Performance Monitoring](#performance-monitoring)
7. [Maintenance Tasks](#maintenance-tasks)
8. [Alerting and Notifications](#alerting-and-notifications)
9. [Troubleshooting](#troubleshooting)

## Monitoring Overview

The system uses a comprehensive monitoring stack:

- **Grafana**: Visualization and dashboards (Port 3001)
- **Prometheus**: Metrics collection and storage (Port 9090)
- **Container Health Checks**: Built-in Docker health monitoring
- **Application Logs**: Centralized logging through Docker

### Key Monitoring Endpoints

| Service | URL | Purpose |
|---------|-----|---------|
| Grafana Dashboard | http://localhost:3001 | System visualization |
| Prometheus UI | http://localhost:9090 | Raw metrics access |
| API Health | http://localhost:8080/health | Service health status |
| Agent Status | http://localhost:8080/api/agents/status | Agent monitoring |

## Grafana Dashboard

### Accessing Grafana

1. Navigate to http://localhost:3001
2. Login with credentials:
   - Username: `admin`
   - Password: `admin` (change in production)
3. Navigate to "Swarm Dashboard"

### Dashboard Panels

#### 1. System Overview Panel
**Metrics Displayed:**
- Overall system health status
- Total number of active services
- Current uptime
- System load average

**Normal Ranges:**
- Health Status: All services "UP"
- Active Services: 7 containers
- Uptime: Continuous since last restart
- Load Average: < 2.0 on 4-core system

#### 2. Service Health Panel
**Metrics Displayed:**
- Individual service status
- Container restart counts
- Health check success rates
- Service response times

**Alert Conditions:**
- Any service showing "DOWN" status
- Restart count > 3 in 1 hour
- Health check failure rate > 5%
- Response time > 1 second

#### 3. Agent Activity Panel
**Metrics Displayed:**
- Agent heartbeat status
- Task queue lengths
- Processing rates
- Agent workload distribution

**Normal Ranges:**
- Heartbeat interval: 30 seconds ± 5 seconds
- Queue length: < 100 pending tasks
- Processing rate: Varies by workload
- Workload: Balanced between agents

#### 4. Resource Utilization Panel
**Metrics Displayed:**
- CPU usage per container
- Memory usage per container
- Disk I/O rates
- Network traffic

**Alert Thresholds:**
- CPU usage > 80% for > 5 minutes
- Memory usage > 85% for any container
- Disk I/O > 1000 IOPS sustained
- Network errors > 1% packet loss

#### 5. Redis Performance Panel
**Metrics Displayed:**
- Redis memory usage
- Commands per second
- Connection count
- Hit/miss ratios

**Normal Ranges:**
- Memory usage: < 80% of allocated
- Commands/sec: Varies by load
- Connections: < 100 concurrent
- Hit ratio: > 90%

### Creating Custom Dashboards

```bash
# Access Grafana API
curl -X POST http://admin:admin@localhost:3001/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @custom-dashboard.json
```

## Prometheus Metrics

### Key Metrics to Monitor

#### System Metrics
```promql
# Container CPU usage
rate(container_cpu_usage_seconds_total[5m])

# Container memory usage
container_memory_usage_bytes / container_spec_memory_limit_bytes

# Container restart count
changes(container_start_time_seconds[1h])
```

#### Application Metrics
```promql
# API request rate
rate(http_requests_total[5m])

# API response time
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Agent heartbeat freshness
time() - agent_last_heartbeat_timestamp
```

#### Redis Metrics
```promql
# Redis memory usage
redis_memory_used_bytes / redis_memory_max_bytes

# Redis command rate
rate(redis_commands_processed_total[5m])

# Redis connection count
redis_connected_clients
```

### Querying Prometheus

```bash
# Direct API queries
curl 'http://localhost:9090/api/v1/query?query=up'

# Query with time range
curl 'http://localhost:9090/api/v1/query_range?query=up&start=2023-01-01T00:00:00Z&end=2023-01-01T01:00:00Z&step=15s'
```

## Health Checks

### Automated Health Checks

The system performs automated health checks:

```bash
# Check all services
./start_system.sh --status

# Individual service checks
docker exec pleiotropy-redis redis-cli ping
curl http://localhost:8080/health
curl http://localhost:3000
```

### Custom Health Check Script

```bash
#!/bin/bash
# health-check.sh

echo "=== System Health Check $(date) ==="

# Check Redis
if docker exec pleiotropy-redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis: HEALTHY"
else
    echo "❌ Redis: UNHEALTHY"
fi

# Check Coordinator API
if curl -f http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Coordinator API: HEALTHY"
else
    echo "❌ Coordinator API: UNHEALTHY"
fi

# Check Web UI
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Web UI: HEALTHY"
else
    echo "❌ Web UI: UNHEALTHY"
fi

# Check Agent Status
AGENT_STATUS=$(curl -s http://localhost:8080/api/agents/status)
if echo "$AGENT_STATUS" | jq -e '.[] | select(.status == "active")' > /dev/null; then
    echo "✅ Agents: ACTIVE"
else
    echo "⚠️  Agents: CHECK REQUIRED"
fi

echo "=== Health Check Complete ==="
```

### Health Check Automation

```bash
# Add to crontab for regular checks
*/5 * * * * /opt/pleiotropy/health-check.sh >> /var/log/pleiotropy-health.log 2>&1
```

## Log Management

### Viewing Logs

```bash
# All services
./start_system.sh --logs

# Specific service
docker-compose logs coordinator
docker-compose logs rust_analyzer
docker-compose logs python_visualizer

# Follow logs in real-time
docker-compose logs -f --tail=100 coordinator

# Filter logs by time
docker-compose logs --since=1h coordinator
docker-compose logs --until=2023-01-01T12:00:00 coordinator
```

### Log Rotation Configuration

```bash
# Configure log rotation
cat > /etc/logrotate.d/pleiotropy << EOF
/var/log/pleiotropy/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    postrotate
        docker-compose restart coordinator
    endscript
}
EOF
```

### Log Analysis

```bash
# Common log analysis commands
grep "ERROR" /var/log/pleiotropy/*.log
grep "agent_heartbeat" /var/log/pleiotropy/*.log | tail -10
awk '/ERROR/ {print $1, $2, $NF}' /var/log/pleiotropy/coordinator.log
```

## Performance Monitoring

### Real-time Performance Monitoring

```bash
# Container resource usage
docker stats

# System resource usage
htop
iostat -x 1
free -h
df -h

# Network monitoring
nethogs
iftop
```

### Performance Benchmarking

```bash
# API performance test
ab -n 1000 -c 10 http://localhost:8080/health

# Redis performance test
docker exec pleiotropy-redis redis-cli --latency-history -i 1

# System load test
stress --cpu 2 --timeout 60s
```

### Performance Alerting Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| CPU Usage | > 70% | > 90% |
| Memory Usage | > 80% | > 95% |
| Disk Usage | > 80% | > 95% |
| API Response Time | > 500ms | > 2s |
| Redis Memory | > 80% | > 95% |

## Maintenance Tasks

### Daily Maintenance

```bash
#!/bin/bash
# daily-maintenance.sh

echo "=== Daily Maintenance $(date) ==="

# 1. Health check
./health-check.sh

# 2. Check disk space
df -h | grep -E "(8[0-9]|9[0-9])%" && echo "⚠️  High disk usage detected"

# 3. Check log file sizes
find /var/log -name "*.log" -size +100M -exec ls -lh {} \;

# 4. Clean up old Docker images
docker image prune -f

# 5. Backup Redis data
docker exec pleiotropy-redis redis-cli BGSAVE

# 6. Export daily metrics
curl -s http://localhost:9090/api/v1/query?query=up > /backup/metrics-$(date +%Y%m%d).json

echo "=== Daily Maintenance Complete ==="
```

### Weekly Maintenance

```bash
#!/bin/bash
# weekly-maintenance.sh

echo "=== Weekly Maintenance $(date) ==="

# 1. Update system metrics baseline
python3 /opt/pleiotropy/scripts/update-baseline-metrics.py

# 2. Archive old trial data
curl -X POST http://localhost:8080/api/trials/archive?days=7

# 3. Clean up old backups
find /backup -name "*.tar.gz" -mtime +30 -delete

# 4. Check for security updates
docker images --format "table {{.Repository}}:{{.Tag}}" | grep -v REPOSITORY

# 5. Performance analysis
python3 /opt/pleiotropy/scripts/weekly-performance-report.py

echo "=== Weekly Maintenance Complete ==="
```

### Monthly Maintenance

```bash
#!/bin/bash
# monthly-maintenance.sh

echo "=== Monthly Maintenance $(date) ==="

# 1. Update Docker images
docker-compose pull
docker-compose up -d

# 2. Full system backup
./scripts/full-backup.sh

# 3. Security audit
./scripts/security-audit.sh

# 4. Performance optimization review
./scripts/performance-optimization.sh

# 5. Documentation update check
git log --since="1 month ago" --oneline | grep -i doc

echo "=== Monthly Maintenance Complete ==="
```

## Alerting and Notifications

### Setting Up Alertmanager (Optional)

```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@pleiotropy.local'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  email_configs:
  - to: 'admin@pleiotropy.local'
    subject: 'Pleiotropy Alert: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
```

### Alert Rules

```yaml
# prometheus-alerts.yml
groups:
- name: pleiotropy
  rules:
  - alert: ServiceDown
    expr: up == 0
    for: 30s
    labels:
      severity: critical
    annotations:
      summary: "Service {{ $labels.instance }} is down"
      
  - alert: HighCPUUsage
    expr: rate(container_cpu_usage_seconds_total[5m]) * 100 > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage on {{ $labels.name }}"
      
  - alert: HighMemoryUsage
    expr: (container_memory_usage_bytes / container_spec_memory_limit_bytes) * 100 > 85
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High memory usage on {{ $labels.name }}"
      
  - alert: AgentHeartbeatMissing
    expr: time() - agent_last_heartbeat_timestamp > 300
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "Agent {{ $labels.agent_name }} heartbeat missing"
```

### Notification Scripts

```bash
#!/bin/bash
# notify-alert.sh

ALERT_TYPE="$1"
ALERT_MESSAGE="$2"
TIMESTAMP=$(date)

# Send email notification
echo "Alert: $ALERT_TYPE at $TIMESTAMP
Message: $ALERT_MESSAGE" | mail -s "Pleiotropy System Alert" admin@example.com

# Send Slack notification (if configured)
curl -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"🚨 Pleiotropy Alert: $ALERT_TYPE\n$ALERT_MESSAGE\"}" \
    "$SLACK_WEBHOOK_URL"

# Log alert
echo "[$TIMESTAMP] ALERT: $ALERT_TYPE - $ALERT_MESSAGE" >> /var/log/pleiotropy-alerts.log
```

## Troubleshooting

### Common Issues and Solutions

#### High Memory Usage
```bash
# Check memory usage by container
docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Restart high-memory containers
docker-compose restart coordinator

# Clean Redis cache if needed
docker exec pleiotropy-redis redis-cli FLUSHDB
```

#### Agent Communication Issues
```bash
# Check agent heartbeats
curl http://localhost:8080/api/agents/status | jq '.[] | {name, last_seen, status}'

# Restart agents
docker-compose restart rust_analyzer python_visualizer

# Check Redis queue
docker exec pleiotropy-redis redis-cli LLEN task_queue
```

#### Slow API Response
```bash
# Check API response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8080/health

# Check system load
uptime
iostat -x 1 5

# Scale services if needed
docker-compose up -d --scale coordinator=2
```

### Monitoring Script Integration

```bash
#!/bin/bash
# comprehensive-monitor.sh

while true; do
    # Health checks
    ./health-check.sh
    
    # Performance checks
    ./performance-check.sh
    
    # Alert checks
    ./alert-check.sh
    
    sleep 300  # Check every 5 minutes
done
```

---

## Monitoring Automation

### Systemd Service for Monitoring

```bash
# /etc/systemd/system/pleiotropy-monitor.service
[Unit]
Description=Pleiotropy Monitoring Service
After=docker.service

[Service]
Type=simple
User=pleiotropy
WorkingDirectory=/opt/pleiotropy
ExecStart=/opt/pleiotropy/comprehensive-monitor.sh
Restart=always

[Install]
WantedBy=multi-user.target
```

### Enable Monitoring Service

```bash
sudo systemctl enable pleiotropy-monitor.service
sudo systemctl start pleiotropy-monitor.service
sudo systemctl status pleiotropy-monitor.service
```

---

*This monitoring guide should be reviewed and updated monthly to ensure all procedures remain current and effective.*