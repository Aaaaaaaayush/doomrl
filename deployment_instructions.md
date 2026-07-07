# Oracle Cloud Deployment Guide — DoomRL Agent

This document outlines the provisioning and configuration workflow to host the **DoomRL Tactical AI Console** on an Always Free Oracle Cloud Infrastructure (OCI) Virtual Machine.

---

## Phase 1: VM Provisioning on Oracle Cloud

1. Log in to the [OCI Console](https://cloud.oracle.com/).
2. Navigate to **Compute** -> **Instances** -> **Create Instance**.
3. **Configure Name & Placement**: Set a name (e.g., `doomrl-vm-ubuntu`) and select your default compartment.
4. **Image and Shape**:
   * **Image**: Click *Edit* -> *Change Image* -> Choose **Canonical Ubuntu 22.04** (do not select 20.04).
   * **Shape**: Select **Ampere VM.Standard.A1.Flex** (ARM64 processor).
   * **Allocation**: Configure **1 OCPU** and **6 GB RAM** (Always Free Eligible).
5. **Networking**:
   * Select/Create a Virtual Cloud Network (VCN) and Subnet.
   * Ensure **Assign a public IPv4 address** is set to **Yes**.
6. **SSH Keys**:
   * Click **Save Private Key** to download the `.key` file. You will need this to access the VM.
7. **Create**: Click **Create** at the bottom and wait for the status to turn green (Running).

---

## Phase 2: Opening Firewall Ports

By default, the Oracle VCN blocks all ports except SSH (22). We must open port 80 (HTTP) and port 443 (HTTPS) to make the dashboard visible to the internet.

1. Under the Instance details page, click on the **Virtual Cloud Network** link.
2. Under the list of Subnets, click on the active **Public Subnet**.
3. Under Security Lists, click on the **Default Security List**.
4. Click **Add Ingress Rules** and configure:
   * **Source Type**: `CIDR`
   * **Source CIDR**: `0.0.0.0/0`
   * **IP Protocol**: `TCP`
   * **Destination Port Range**: `80,443`
   * **Description**: `Allow web traffic (HTTP/HTTPS)`
5. Click **Add Ingress Rules**.

---

## Phase 3: Domain Setup (DuckDNS)

We will configure a free subdomain (e.g., `aayush-doomrl.duckdns.org`) so users do not have to type the raw IP address and to allow Let's Encrypt to issue SSL certificates.

1. Go to [DuckDNS](https://www.duckdns.org/) and log in using any auth provider.
2. In the *domains* section, type your target domain name (e.g., `aayush-doomrl`) and click **add domain**.
3. In the *ip* box next to your subdomain, enter the **Public IP Address** of your Oracle Cloud VM and click **update ip**.
4. Leave the tab open to copy the **Token** for the auto-renewal configuration.

---

## Phase 3.5: Push Local Code to GitHub

Before setting up the environment on the VM, you must initialize Git locally and push your code to GitHub. Run these commands on your **local machine** inside `d:\Projects_Msc\doomrl`:

```powershell
# 1. Initialize git repository
git init

# 2. Stage files (ignores weights, videos, and venv due to .gitignore)
git add .

# 3. Commit files
git commit -m "Initial commit: DoomRL Agent and HUD Dashboard"

# 4. Set branch to main
git branch -M main

# 5. Link to your GitHub remote (create a repo named 'doomrl' on GitHub first)
git remote add origin https://github.com/yourusername/doomrl.git

# 6. Push files
git push -u origin main
```

---

## Phase 4: Server Environment Setup

Now, connect to the instance via SSH and install the system dependencies:

```bash
# 1. Connect to the VM (replace with your private key path and Oracle public IP)
ssh -i "C:\Users\Aayush\Downloads\ssh-key-2026-07-07.key" ubuntu@130.210.63.150

# 2. Update system repository and packages
sudo apt update && sudo apt upgrade -y

# 3. Install core dependencies
sudo apt install -y python3.10 python3.10-venv python3-pip nginx git certbot python3-certbot-nginx

# 4. Clone the repository
git clone https://github.com/yourusername/doomrl.git
cd doomrl

# 5. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 6. Install dependencies (CPU-only configuration)
# Since the VM only serves pre-recorded videos and JSON stats, CPU-only PyTorch is fine.
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## Phase 5: Uploading Local Assets to the VM

Since the VM does not train the models, we must upload the trained checkpoints and recorded videos from our local computer to the VM. Run this on your **local development machine** in a separate terminal:

```powershell
# Run from inside the local doomrl/ folder:
# Upload trained model checkpoints
scp -i path/to/your/ssh_key.key -r models/ ubuntu@<oracle-vm-ip>:/home/ubuntu/doomrl/

# Upload recorded videos
scp -i path/to/your/ssh_key.key -r videos/ ubuntu@<oracle-vm-ip>:/home/ubuntu/doomrl/

# Upload JSON export metrics
scp -i path/to/your/ssh_key.key -r logs/ ubuntu@<oracle-vm-ip>:/home/ubuntu/doomrl/
```

---

## Phase 6: Nginx Reverse Proxy Configuration

Create an Nginx configuration file to redirect external HTTP/HTTPS traffic to the Uvicorn backend:

1. Open a new configuration file:
   ```bash
   sudo nano /etc/nginx/sites-available/doomrl
   ```
2. Paste the following configuration (replace `your-subdomain` with your DuckDNS subdomain):
   ```nginx
   server {
       listen 80;
       server_name your-subdomain.duckdns.org;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```
3. Enable the site configuration and restart Nginx:
   ```bash
   sudo ln -s /etc/nginx/sites-available/doomrl /etc/nginx/sites-enabled/
   sudo rm /etc/nginx/sites-enabled/default
   sudo nginx -t
   sudo systemctl restart nginx
   ```

---

## Phase 7: SSL Certificate (Let's Encrypt)

Obtain a free SSL certificate to secure the connection:

```bash
sudo certbot --nginx -d your-subdomain.duckdns.org
```

Certbot will automatically verify the domain, obtain the certificate, and update the Nginx configuration. Select **2** if asked whether to redirect all HTTP traffic to HTTPS.

---

## Phase 8: Systemd Service Configuration

Configure a systemd service to keep the FastAPI server running in the background:

1. Create a service file:
   ```bash
   sudo nano /etc/systemd/system/doomrl.service
   ```
2. Paste the following content:
   ```ini
   [Unit]
   Description=FastAPI DoomRL Dashboard Server
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/doomrl
   ExecStart=/home/ubuntu/doomrl/.venv/bin/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```
3. Start the service and enable it to run at boot:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl start doomrl
   sudo systemctl enable doomrl
   ```

To check the server logs, use:
```bash
sudo journalctl -u doomrl -f
```

---

## Phase 9: DuckDNS Cron Auto-Update

To ensure the DuckDNS subdomain always tracks the VM's public IP (which can change if the VM restarts):

1. Create a DuckDNS directory and update script:
   ```bash
   mkdir ~/duckdns
   nano ~/duckdns/duck.sh
   ```
2. Paste the following update command (replace `<your-subdomain>` and `<your-token>`):
   ```bash
   echo url="https://www.duckdns.org/update?domains=<your-subdomain>&token=<your-token>&ip=" | curl -k -o ~/duckdns/duck.log -K -
   ```
3. Set execution permissions and add to the system cron:
   ```bash
   chmod +x ~/duckdns/duck.sh
   crontab -e
   ```
4. Append this line at the bottom of the crontab to run the script every 5 minutes:
   ```cron
   */5 * * * * ~/duckdns/duck.sh >/dev/null 2>&1
   ```
