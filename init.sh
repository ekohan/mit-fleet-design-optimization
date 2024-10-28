#!/bin/bash

set -e  # Exit on error

# Create Python virtual environment
echo "Creating Python virtual environment..."
python -m venv mit-fleet-env

# Activate the virtual environment
echo "Activating Python virtual environment..."
source mit-fleet-env/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Run the import script to create the database
echo "Creating SQLite database..."
python data/import.py

# All done!
echo "Project setup complete!"