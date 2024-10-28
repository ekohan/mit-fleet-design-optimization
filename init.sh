#!/bin/bash

echo "Eric is great!"
set -e  # Exit on error

# Check if the virtual environment already exists
if [ ! -d "mit-fleet-env" ]; then
  echo "Creating Python virtual environment..."
  python -m venv mit-fleet-env
else
  echo "Virtual environment 'mit-fleet-env' already exists. Skipping creation."
fi

# Activate the virtual environment
echo "Activating Python virtual environment..."

# Check the OS and set the activation command accordingly
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  # For Windows
  source mit-fleet-env/Scripts/activate
else
  # For Unix-based systems (Linux, macOS)
  source mit-fleet-env/bin/activate
fi

# Verify activation
if [[ "$VIRTUAL_ENV" != "" ]]; then
  echo "Virtual environment activated."
else
  echo "Failed to activate virtual environment. Exiting."
  exit 1
fi

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Run the import script to create the database
echo "Creating SQLite database..."
python data/import.py

# All done!
echo "Project setup complete!"
