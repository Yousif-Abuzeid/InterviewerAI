#!/bin/bash

# Prompt user to enter the values securely
read -p "Enter your OpenAI API Key: " OPENAI_API_KEY
read -p "Enter your Gmail address: " GMAIL_ADDRESS
read -p "Enter your Gmail app password: " GMAIL_APP_PASSWORD
read -p "Enter the recipient email: " RECIPIENT_EMAIL

# Export them as environment variables
export OPENAI_API_KEY="$OPENAI_API_KEY"
export GMAIL_ADDRESS="$GMAIL_ADDRESS_KEY"
export GMAIL_APP_PASSWORD="$GMAIL_APP_PASSWORD_KEY"
export RECIPIENT_EMAIL="$RECIPIENT_EMAIL_KEY"

echo "Environment variables have been set for this session."
