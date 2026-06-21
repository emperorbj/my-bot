#!/bin/bash
echo "initializing package manager🚀🚀🚀"
uv init
echo "setting up virtual environment🐍🐍🐍"
uv venv

echo "installing dependencies📦📦📦"
uv pip sync requirements.txt