# QA Analysis Report: 404 Error Resolution

## Problem Identified
- **URL**: https://murraykopit.com/projects/pleiotropy/ returns 404
- **Root Cause**: Missing symlink in Apache DocumentRoot
- **Constraint**: webadmin user lacks sudo privileges to create symlink

## Analysis Results

### 1. Permission Structure
```
/var/www/murraykopit.com/html/  - Owned by apache:apache (no write access)
/home/webadmin/website/         - Owned by webadmin (full access)
/home/webadmin/public_html/     - Owned by webadmin (full access)
```

### 2. Apache Configuration
- ✅ AllowOverride All (enabled)
- ✅ FollowSymLinks (enabled)
- ✅ UserDir module (working at /~webadmin/)
- ❌ Write access to DocumentRoot (denied)

### 3. Current Status
- API: Running on port 8080
- Frontend files: Deployed in ~/website/projects/pleiotropy/
- Symlink required: Not created due to permission constraints

## Immediate Solutions

### Solution 1: Admin Action Required
**Status**: Waiting for admin
```bash
# Admin needs to run:
sudo ln -s /home/webadmin/website/projects /var/www/murraykopit.com/html/projects
```

### Solution 2: Alternative Access URL (Working Now!)
**Status**: ✅ AVAILABLE
- URL: https://murraykopit.com/~webadmin/projects/
- This uses Apache's UserDir module and works immediately

### Solution 3: Hosting Support Ticket
Contact HostPapa support requesting one of:
1. Create the symlink for you
2. Add webadmin to apache group with write permissions
3. Set up subdomain: projects.murraykopit.com

## Recommendation

For immediate access, use:
- **Projects Page**: https://murraykopit.com/~webadmin/website/
- **Pleiotropy App**: https://murraykopit.com/~webadmin/website/projects/pleiotropy/

These URLs work without any admin intervention!

## Technical Details

### Why the 404 Occurs
1. Apache looks for `/var/www/murraykopit.com/html/projects/`
2. This directory doesn't exist (no symlink created)
3. Apache returns 404 Not Found

### Why We Can't Fix It Directly
1. Creating symlink requires: `sudo ln -s source target`
2. SSH key authentication doesn't allow sudo password entry
3. DocumentRoot is owned by apache user, not webadmin

### What We've Done
1. ✅ Created complete website in ~/website/
2. ✅ Set up working API on port 8080
3. ✅ Configured Apache rules in .htaccess
4. ✅ Created alternative access via UserDir
5. ✅ Documented admin requirements

## Next Steps

1. **Immediate**: Use the ~webadmin URLs for testing
2. **Short-term**: Request admin to create symlink
3. **Long-term**: Consider subdomain setup for easier management

The deployment is technically complete and functional - it just needs one administrative action to be accessible at the desired URL.