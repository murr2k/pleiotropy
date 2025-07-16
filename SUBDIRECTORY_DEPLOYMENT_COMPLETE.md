# Pleiotropy Subdirectory Deployment - COMPLETE ✅

## Summary

The pleiotropy website has been successfully deployed as a subdirectory installation with a tabbed projects homepage. All components are ready and the API is running.

## What Was Created

### 1. Main Projects Page (`~/website/`)
- **URL**: `https://murraykopit.com/projects/`
- **Features**:
  - Professional tabbed interface with 4 project tabs
  - Pleiotropy (active and functional)
  - Quantum Computing (placeholder)
  - AI Research (placeholder)
  - Cryptography (placeholder)
- **Files**:
  - `index.html` - Main projects page with Bootstrap tabs
  - `style.css` - Custom styling
  - `.htaccess` - Apache rewrite rules

### 2. Pleiotropy Application (`~/website/projects/pleiotropy/`)
- **URL**: `https://murraykopit.com/projects/pleiotropy/`
- **Features**:
  - Full genomic analysis interface
  - File upload capability
  - Real-time results display
  - API integration configured for subdirectory
- **Files**:
  - `index.html` - React-based pleiotropy app
  - `style.css` - Application styles

### 3. Backend API
- **Status**: ✅ RUNNING on port 8080
- **Endpoints**:
  - `/api/health` - Health check
  - `/api/info` - API information
  - `/api/analyze` - File analysis
  - `/api/analyses` - List analyses
- **Process**: Running as background process with logs at `~/pleiotropy/logs/api.log`

## Directory Structure
```
/home/webadmin/
├── website/                    # Web root
│   ├── index.html             # Main projects page
│   ├── style.css              # Main styles
│   ├── .htaccess              # Apache rules
│   ├── ADMIN_INSTRUCTIONS.txt # Setup guide
│   └── projects/
│       ├── pleiotropy/        # Pleiotropy app
│       │   ├── index.html
│       │   └── style.css
│       ├── project2/          # Future project
│       └── project3/          # Future project
│
└── pleiotropy/                # Backend
    ├── api_server.py          # FastAPI server
    ├── venv/                  # Python environment
    ├── data/                  # SQLite database
    └── logs/                  # API logs
```

## Admin Actions Required

The server administrator needs to complete ONE simple step:

### Create Symbolic Link
```bash
sudo ln -s /home/webadmin/website /var/www/murraykopit.com/html/projects
```

### Optional: Configure Apache for API Proxy
If the .htaccess rules don't work, add to Apache config:
```apache
<Location /projects/pleiotropy/api>
    ProxyPass http://localhost:8080/api
    ProxyPassReverse http://localhost:8080/api
</Location>
```

## Testing

Once the admin creates the symlink:

1. **Test Projects Page**:
   ```
   https://murraykopit.com/projects/
   ```

2. **Test Pleiotropy App**:
   ```
   https://murraykopit.com/projects/pleiotropy/
   ```

3. **Test API** (should return JSON):
   ```
   https://murraykopit.com/projects/pleiotropy/api/health
   ```

## Features Implemented

✅ **Multi-project homepage** with professional tabbed interface
✅ **Pleiotropy app** modified for subdirectory deployment
✅ **API running** and configured for proxy access
✅ **Clean URLs** with proper navigation
✅ **Back button** to return to projects page
✅ **Responsive design** for mobile and desktop
✅ **Demo mode** if API is unavailable

## Maintenance

- **View API logs**: `tail -f ~/pleiotropy/logs/api.log`
- **Restart API**: `pkill -f api_server.py && cd ~/pleiotropy && ./deploy_subdirectory.sh`
- **Add new projects**: Create directories in `~/website/projects/`
- **Update content**: Edit files directly in `~/website/`

## Next Steps

1. Admin creates the symbolic link
2. Test all functionality
3. Add content for other project tabs as needed
4. Consider setting up systemd service for API persistence

The deployment is **100% complete** from the webadmin side!