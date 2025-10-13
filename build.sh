#!/bin/bash
# Build script for WhatNow backend on Render
# Optimized for Python 3.13 compatibility

echo "🔧 Setting up build environment for Python 3.13..."

# Upgrade pip and build tools first
pip install --upgrade pip setuptools wheel

# Install build dependencies
pip install --upgrade build

# Install requirements with no cache to avoid conflicts
pip install --no-cache-dir -r requirements.txt

echo "✅ Build completed successfully!"
