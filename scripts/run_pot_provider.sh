#!/bin/bash
# Ensure the bgutil PO token provider is built, then run it (HTTP mode, port 4416).
# Self-healing: builds the server first if it is missing (e.g. on a fresh env).
set -e

bash scripts/setup_pot_provider.sh

cd vendor/bgutil-pot/server
exec node build/main.js --port 4416
