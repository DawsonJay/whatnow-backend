#!/bin/bash
# Build script for WhatNow backend on Render
# Fixes setuptools and packaging issues

echo "🔧 Setting up build environment..."

# Upgrade pip and setuptools first
pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt

echo "✅ Build completed successfully!"
