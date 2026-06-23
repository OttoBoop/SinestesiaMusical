#!/bin/bash
# Runs automatically after a task is merged. Installs Python deps and rebuilds
# the bgutil PO token provider so YouTube downloads work without manual setup.
set -e

uv sync

bash scripts/setup_pot_provider.sh
