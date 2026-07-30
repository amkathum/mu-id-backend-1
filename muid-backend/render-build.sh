#!/usr/bin/env bash
set -e

apt-get update && apt-get install -y ffmpeg
pip install -r requirements.txt
