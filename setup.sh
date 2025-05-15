#!/bin/bash

# set_env.sh
# This script prompts the user to enter environment variables and sets them for the current session.
# It also creates a .env file for persistent use with python-dotenv.

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Function to print messages
print_message() {
    echo -e "${GREEN}[INFO] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# Prompt user to enter the values securely
echo "Enter your environment variables (press Enter to skip if setting manually later):"
read -p "OpenAI API Key: " OPENAI_API_KEY
read -p "Gmail Address: " GMAIL_ADDRESS
read -p "Gmail App Password: " GMAIL_APP_PASSWORD
read -p "Recipient Email: " RECIPIENT_EMAIL

# Validate inputs
if [ -z "$OPENAI_API_KEY" ] || [ -z "$GMAIL_ADDRESS" ] || [ -z "$GMAIL_APP_PASSWORD" ] || [ -z "$RECIPIENT_EMAIL" ]; then
    print_message "Some variables were not provided. You can set them manually in a .env file or run this script again."
else
    # Export them as environment variables for the current session
    export OPENAI_API_KEY="$OPENAI_API_KEY"
    export GMAIL_ADDRESS_KEY="$GMAIL_ADDRESS"
    export GMAIL_APP_PASSWORD_KEY="$GMAIL_APP_PASSWORD"
    export RECIPIENT_EMAIL_KEY="$RECIPIENT_EMAIL"
    print_message "Environment variables set for this session."

    # Save to .env file for persistent use
    cat << EOF > .env
OPENAI_API_KEY=$OPENAI_API_KEY
GMAIL_ADDRESS_KEY=$GMAIL_ADDRESS
GMAIL_APP_PASSWORD_KEY=$GMAIL_APP_PASSWORD
RECIPIENT_EMAIL_KEY=$RECIPIENT_EMAIL
EOF
    print_message ".env file created successfully."
fi

echo
echo "============================================================="
echo "To use these variables in your script:"
echo "1. Ensure the virtual environment is activated:"
echo "   source venv/bin/activate"
echo "2. Run the Python script:"
echo "   python3 conductor.py"
echo "3. If you skipped any inputs, create a .env file manually or run this script again."
echo "============================================================="