#!/usr/bin/env bash
# deploy_beta.sh — HEATER beta deployment helper
# Usage: ./scripts/deploy_beta.sh [cloud|vps]
#   cloud  — Streamlit Cloud deployment instructions (default)
#   vps    — Self-hosted VPS deployment commands

set -euo pipefail

MODE="${1:-cloud}"

print_header() {
    echo ""
    echo "============================================"
    echo "  HEATER Beta Deployment — ${1}"
    echo "============================================"
    echo ""
}

deploy_cloud() {
    print_header "Streamlit Cloud"

    echo "STEP 1: Push your code to GitHub"
    echo "---------------------------------"
    echo "  git add ."
    echo "  git commit -m 'chore: prepare for Streamlit Cloud deployment'"
    echo "  git push origin master"
    echo ""

    echo "STEP 2: Connect repo at share.streamlit.io"
    echo "-------------------------------------------"
    echo "  1. Go to https://share.streamlit.io"
    echo "  2. Sign in with your GitHub account (hicklax13)"
    echo "  3. Click 'New app'"
    echo "  4. Repository:  hicklax13/HEATER_v1.0.0"
    echo "  5. Branch:      master"
    echo "  6. Main file:   app.py"
    echo "  7. Click 'Deploy'"
    echo ""

    echo "STEP 3: Configure secrets"
    echo "--------------------------"
    echo "  In the Streamlit Cloud dashboard:"
    echo "  1. Open your app → Settings → Secrets"
    echo "  2. Paste the contents of .streamlit/secrets.toml.example"
    echo "     with your actual Yahoo credentials filled in:"
    echo ""
    echo "     [yahoo]"
    echo "     consumer_key    = \"<your Yahoo consumer key>\""
    echo "     consumer_secret = \"<your Yahoo consumer secret>\""
    echo ""
    echo "  3. Click 'Save' — the app will automatically restart."
    echo ""

    echo "STEP 4: Verify deployment"
    echo "--------------------------"
    echo "  - Watch the build log for pip install errors."
    echo "  - Confirm the app loads the bootstrap screen."
    echo "  - Test Yahoo OAuth reconnect via the sidebar."
    echo ""

    echo "NOTES"
    echo "-----"
    echo "  - Python version: set in packages.txt or via Advanced Settings (use 3.11)."
    echo "  - requirements.txt must be committed (it is)."
    echo "  - data/draft_tool.db is NOT committed; bootstrap recreates it on first run."
    echo "  - data/yahoo_token.json is NOT committed; OAuth flow writes it at runtime."
    echo ""
}

deploy_vps() {
    print_header "VPS (Self-Hosted)"

    APP_DIR="/opt/heater"
    REPO="git@github.com:hicklax13/HEATER_v1.0.0.git"
    PORT=8501

    echo "STEP 1: SSH into your server and clone the repo"
    echo "-------------------------------------------------"
    echo "  ssh user@your-server-ip"
    echo "  sudo mkdir -p ${APP_DIR}"
    echo "  sudo chown \$USER:\$USER ${APP_DIR}"
    echo "  git clone ${REPO} ${APP_DIR}"
    echo "  cd ${APP_DIR}"
    echo ""

    echo "STEP 2: Create and activate a virtual environment"
    echo "--------------------------------------------------"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo ""

    echo "STEP 3: Install dependencies"
    echo "-----------------------------"
    echo "  pip install --upgrade pip"
    echo "  pip install -r requirements.txt"
    echo ""

    echo "STEP 4: Copy secrets into place"
    echo "--------------------------------"
    echo "  # From your local machine, copy the real secrets file:"
    echo "  scp .streamlit/secrets.toml user@your-server-ip:${APP_DIR}/.streamlit/secrets.toml"
    echo ""
    echo "  # Or create it directly on the server:"
    echo "  nano ${APP_DIR}/.streamlit/secrets.toml"
    echo "  # Paste your credentials, save, exit."
    echo ""

    echo "STEP 5: Run Streamlit with nohup (persists after SSH disconnect)"
    echo "-----------------------------------------------------------------"
    echo "  cd ${APP_DIR}"
    echo "  source venv/bin/activate"
    echo "  nohup streamlit run app.py \\"
    echo "      --server.port ${PORT} \\"
    echo "      --server.headless true \\"
    echo "      --server.enableCORS false \\"
    echo "      > logs/streamlit.log 2>&1 &"
    echo "  echo \"Streamlit PID: \$!\""
    echo ""
    echo "  # To stop: kill \$(lsof -ti:${PORT})"
    echo ""

    echo "STEP 6: Nginx reverse proxy (recommended)"
    echo "------------------------------------------"
    echo "  # Install nginx if needed: sudo apt install nginx"
    echo "  # Create /etc/nginx/sites-available/heater with:"
    echo ""
    echo "  server {"
    echo "      listen 80;"
    echo "      server_name your-domain.com;"
    echo ""
    echo "      location / {"
    echo "          proxy_pass         http://localhost:${PORT};"
    echo "          proxy_http_version 1.1;"
    echo "          proxy_set_header   Upgrade \$http_upgrade;"
    echo "          proxy_set_header   Connection 'upgrade';"
    echo "          proxy_set_header   Host \$host;"
    echo "          proxy_cache_bypass \$http_upgrade;"
    echo "      }"
    echo "  }"
    echo ""
    echo "  sudo ln -s /etc/nginx/sites-available/heater /etc/nginx/sites-enabled/"
    echo "  sudo nginx -t && sudo systemctl reload nginx"
    echo ""

    echo "STEP 7: Verify deployment"
    echo "--------------------------"
    echo "  curl -I http://localhost:${PORT}  # Should return HTTP 200"
    echo "  tail -f ${APP_DIR}/logs/streamlit.log"
    echo ""

    echo "NOTES"
    echo "-----"
    echo "  - Python 3.11+ required (3.14 for local dev, 3.11 recommended for VPS stability)."
    echo "  - Create ${APP_DIR}/data/ and ${APP_DIR}/logs/ directories before first run."
    echo "  - For HTTPS, use Certbot: sudo certbot --nginx -d your-domain.com"
    echo "  - Consider systemd service for auto-restart on server reboot."
    echo ""
}

case "${MODE}" in
    cloud)
        deploy_cloud
        ;;
    vps)
        deploy_vps
        ;;
    *)
        echo "ERROR: Unknown mode '${MODE}'. Use 'cloud' or 'vps'." >&2
        echo "Usage: $0 [cloud|vps]" >&2
        exit 1
        ;;
esac
