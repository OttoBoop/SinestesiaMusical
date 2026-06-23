#!/bin/bash
# Idempotently set up the bgutil Proof-of-Origin (PO) token provider server.
# This companion service lets yt-dlp satisfy YouTube's "confirm you're not a
# bot" challenge automatically, with no account or login. Safe to run repeatedly.
set -e

DIR="vendor/bgutil-pot"
# Keep this in lockstep with the bgutil-ytdlp-pot-provider Python plugin version.
VERSION="1.3.1"

if [ -f "$DIR/server/build/main.js" ] && [ -d "$DIR/server/node_modules" ]; then
    echo "bgutil PO provider already built."
    exit 0
fi

echo "Setting up bgutil PO token provider ($VERSION)..."
rm -rf "$DIR"
mkdir -p vendor
git clone --depth 1 --single-branch --branch "$VERSION" \
    https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git "$DIR"
(
    cd "$DIR/server"
    npm ci
    npx tsc
)
echo "bgutil PO provider ready."
