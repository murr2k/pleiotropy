# Deployment Verification Checklist

This document provides a comprehensive checklist for verifying successful deployment of the Genomic Pleiotropy Cryptanalysis system.

## Pre-Deployment Requirements

### Environment Setup
- [ ] Docker 20.10+ installed and running
- [ ] Docker Compose 1.29+ installed
- [ ] Git installed and repository cloned
- [ ] Sufficient system resources (4+ CPU cores, 8GB+ RAM, 50GB+ storage)
- [ ] Network ports available (3000, 8080, 3001, 6379, 9090)

### Security Configuration
- [ ] Firewall rules configured for required ports
- [ ] SSL certificates prepared (for production deployments)
- [ ] Admin passwords changed from defaults
- [ ] Network access restrictions configured

## Deployment Steps

### 1. Initial Deployment
```bash
# Clone repository
git clone https://github.com/murr2k/pleiotropy.git
cd pleiotropy

# Start system
./start_system.sh --docker -d
```
- [ ] Repository cloned successfully
- [ ] Docker images built without errors
- [ ] All services started in detached mode

### 2. Service Health Verification
```bash
# Check service status
./start_system.sh --status
```
- [ ] All containers running (`docker ps` shows 7 containers)
- [ ] Health checks passing for all services
- [ ] No error logs in startup phase

### 3. Individual Service Tests

#### Redis Cache
```bash
docker exec pleiotropy-redis redis-cli ping
```
- [ ] Returns "PONG"
- [ ] Redis container healthy and responding

#### Coordinator API
```bash
curl -f http://localhost:8080/health
```
- [ ] Returns `{"status": "healthy"}`
- [ ] API endpoints accessible
- [ ] OpenAPI documentation available at http://localhost:8080/docs

#### Web UI
```bash
curl -f http://localhost:3000
```
- [ ] Web UI loads successfully
- [ ] React application renders without errors
- [ ] Dashboard components display correctly

#### Monitoring Services
```bash
curl -f http://localhost:3001  # Grafana
curl -f http://localhost:9090  # Prometheus
```
- [ ] Grafana dashboard accessible (admin/admin)
- [ ] Prometheus metrics endpoint responding
- [ ] Swarm dashboard configured and displaying data

### 4. Agent Communication Verification
```bash
curl http://localhost:8080/api/agents/status | jq
```
- [ ] Both agents (rust_analyzer, python_visualizer) reporting
- [ ] Heartbeat timestamps recent (< 60 seconds old)
- [ ] Agent status shows "active" or "idle"
- [ ] No communication errors in logs

### 5. Data Persistence Verification
```bash
docker volume ls | grep pleiotropy
```
- [ ] Redis data volume exists and mounted
- [ ] Prometheus data volume exists and mounted
- [ ] Grafana data volume exists and mounted
- [ ] Reports directory accessible and writable

## Functional Testing

### 6. End-to-End Workflow Test
```bash
# Submit a test analysis (if test data available)
curl -X POST http://localhost:8080/api/trials/analyze \
  -H "Content-Type: application/json" \
  -d '{"genome_file": "test_ecoli.fasta", "organism": "ecoli"}'
```
- [ ] Analysis request accepted
- [ ] Task distributed to appropriate agent
- [ ] Processing begins without errors
- [ ] Results stored in database
- [ ] Progress visible in dashboard

### 7. Monitoring and Alerting
```bash
# Check Grafana dashboards
# Navigate to http://localhost:3001
```
- [ ] System overview dashboard displays all services
- [ ] Agent activity panel shows current workload
- [ ] Performance metrics panel shows resource usage
- [ ] No critical alerts or error indicators

### 8. Load Testing (Optional)
```bash
# Basic load test
ab -n 100 -c 10 http://localhost:8080/health
```
- [ ] API handles concurrent requests successfully
- [ ] Response times remain under 200ms
- [ ] No memory leaks or resource exhaustion
- [ ] System remains stable under load

## Security Verification

### 9. Network Security
```bash
# Check open ports
netstat -tlnp | grep -E '(3000|8080|3001|6379|9090)'
```
- [ ] Only required ports are open
- [ ] Services bound to appropriate interfaces
- [ ] Firewall rules properly configured

### 10. Container Security
```bash
# Check container users
docker exec pleiotropy-coordinator id
docker exec pleiotropy-rust-analyzer id
```
- [ ] Containers running as non-root users
- [ ] File permissions properly configured
- [ ] No unnecessary capabilities granted

## Backup and Recovery

### 11. Backup Functionality
```bash
# Test backup creation
docker exec pleiotropy-redis redis-cli BGSAVE
docker run --rm -v pleiotropy_redis_data:/data -v $(pwd):/backup alpine tar czf /backup/test-backup.tar.gz /data
```
- [ ] Redis backup completes successfully
- [ ] Volume backup creates valid archive
- [ ] Backup files accessible and readable

### 12. Recovery Testing
```bash
# Test recovery procedure (in test environment only)
./start_system.sh --stop
docker volume rm pleiotropy_redis_data
docker-compose up -d redis
# Restore from backup...
```
- [ ] Recovery procedure documented and tested
- [ ] Data restoration works correctly
- [ ] Services restart after recovery

## Performance Benchmarks

### 13. Resource Usage Baselines
```bash
docker stats --no-stream
```
- [ ] Memory usage within expected limits
- [ ] CPU utilization reasonable at idle
- [ ] Network I/O patterns normal
- [ ] Disk usage tracking implemented

### 14. Response Time Baselines
- [ ] API health endpoint: < 50ms
- [ ] Dashboard load time: < 2 seconds
- [ ] Agent heartbeat interval: 30 seconds
- [ ] Monitoring update frequency: 15 seconds

## Documentation Verification

### 15. Documentation Completeness
- [ ] README.md updated with deployment instructions
- [ ] CLAUDE.md contains operational procedures
- [ ] API documentation accessible and current
- [ ] Troubleshooting guides available
- [ ] Monitoring setup documented

### 16. User Access Instructions
- [ ] Service URLs documented and accessible
- [ ] Login credentials provided where needed
- [ ] User guides available for each interface
- [ ] Support contact information provided

## Post-Deployment Configuration

### 17. Production Hardening
- [ ] Default passwords changed
- [ ] Log rotation configured
- [ ] Monitoring alerts configured
- [ ] Backup automation scheduled
- [ ] Security updates planned

### 18. Operational Procedures
- [ ] Daily health check procedures documented
- [ ] Emergency response procedures defined
- [ ] Maintenance windows scheduled
- [ ] Update procedures documented

## Sign-off Checklist

### Technical Validation
- [ ] All services deployed and operational
- [ ] Health checks passing
- [ ] Monitoring functional
- [ ] Backup/recovery tested
- [ ] Performance meets requirements

### Documentation
- [ ] User guides complete
- [ ] Operational procedures documented
- [ ] Troubleshooting guides available
- [ ] Contact information current

### Security
- [ ] Security configuration verified
- [ ] Access controls implemented
- [ ] Network security configured
- [ ] Audit trail enabled

### Support Readiness
- [ ] Monitoring dashboards configured
- [ ] Alert thresholds set
- [ ] Support team trained
- [ ] Escalation procedures defined

---

## Deployment Sign-off

**Deployment Date:** _______________

**System Administrator:** _______________

**Technical Lead:** _______________

**Security Review:** _______________

**Final Approval:** _______________

**Notes:**
```
[Space for deployment-specific notes and observations]
```

**Post-Deployment Follow-up:**
- [ ] 24-hour stability check scheduled
- [ ] Weekly performance review scheduled
- [ ] Monthly maintenance review scheduled
- [ ] Quarterly security review scheduled

---

*This checklist should be completed for every deployment and kept as documentation for operational reference.*