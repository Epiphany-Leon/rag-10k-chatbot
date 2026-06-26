# Deploying 10-K RAG Studio

Two ways to run it on a server. Both expose the app on port 80 via nginx.

## Option A — automated (Ubuntu/Debian)

SSH into the server, then:

```bash
REPO=https://github.com/Epiphany-Leon/rag-10k-chatbot.git \
DOMAIN=<your-server-ip-or-domain> \
bash <(curl -fsSL https://raw.githubusercontent.com/Epiphany-Leon/rag-10k-chatbot/main/deploy/deploy.sh)
```

The script installs dependencies, creates a venv, sets up a `systemd` service
and an nginx reverse proxy, and starts everything. On first run it creates
`.env` from the template — **edit `~/rag-10k-chatbot/.env` to add your API keys,
then re-run the script** (it is idempotent).

## Option B — manual

```bash
sudo apt-get update && sudo apt-get install -y python3-venv git nginx
git clone https://github.com/Epiphany-Leon/rag-10k-chatbot.git
cd rag-10k-chatbot
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cp .env.example .env && nano .env            # add keys
./.venv/bin/python scripts/build_index.py    # pre-build index

# systemd
sudo cp deploy/ragstudio.service /etc/systemd/system/
sudo sed -i "s|__USER__|$(whoami)|g; s|__APP_DIR__|$(pwd)|g" /etc/systemd/system/ragstudio.service
sudo systemctl daemon-reload && sudo systemctl enable --now ragstudio

# nginx
sudo cp deploy/nginx.conf /etc/nginx/sites-available/ragstudio
sudo sed -i "s|__DOMAIN_OR_IP__|<your-ip>|g" /etc/nginx/sites-available/ragstudio
sudo ln -sf /etc/nginx/sites-available/ragstudio /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

## Operations

```bash
sudo systemctl status ragstudio      # health
sudo journalctl -u ragstudio -f      # live logs
sudo systemctl restart ragstudio     # restart after a code/.env change
```

## Notes

- **Open port 80** (and 22) in the cloud provider's firewall / security group.
- **HTTPS:** add a domain and run `sudo certbot --nginx` for a free Let's Encrypt
  certificate.
- **Docker alternative:** the app is a plain Streamlit process, so a 3-line
  Dockerfile (`FROM python:3.11-slim`, install requirements, `CMD streamlit run
  app.py`) also works if you prefer containers.
- The local embeddings are called via API, so the server needs **no GPU** and
  modest RAM (~1 GB).
