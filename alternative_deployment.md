# Alternative Deployment Solution

## Problem
The main issue is that creating a symlink in `/var/www/murraykopit.com/html/` requires sudo privileges.

## Solution: Use .htaccess Redirect

Since we can't create the symlink without sudo, we'll use Apache's mod_rewrite to redirect requests.

### Option 1: Redirect to UserDir (if UserDir is enabled on the domain)
Create `/var/www/murraykopit.com/html/.htaccess` with:
```apache
RewriteEngine On
RewriteRule ^projects/(.*)$ https://murraykopit.com/~webadmin/projects/$1 [L,R=301]
```

### Option 2: Copy files directly (requires write access)
If we can get write access to the DocumentRoot, we can copy the files instead of symlinking.

### Option 3: Use existing accessible directory
Since `~/public_html` seems to work with UserDir, we can:
1. Move all content to `~/public_html/projects/`
2. Access via `https://murraykopit.com/~webadmin/projects/`

### Option 4: Request hosting support
Contact HostPapa support to:
1. Create the symlink: `ln -s /home/webadmin/website/projects /var/www/murraykopit.com/html/projects`
2. Or grant write permissions to the apache group
3. Or enable a subdomain like `projects.murraykopit.com`