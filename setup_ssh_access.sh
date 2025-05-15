#!/bin/bash

#############################################################
# SSH Key Management Script for Interview System
#############################################################
#
# USAGE:
#   bash setup_interview.sh USERNAME GMAIL_ADDRESS GMAIL_APP_PASSWORD
#
# ARGUMENTS:
#   USERNAME         - Username for the system account (e.g., salma)
#   GMAIL_ADDRESS    - Gmail address to send instructions to participant
#   GMAIL_APP_PASSWORD - Gmail app password for sending emails
#
# EXAMPLE:
#   bash setup_interview.sh salma example@gmail.com abcd efgh ijkl mnop
#
# DESCRIPTION:
#   This script sets up SSH access for an interview participant with automatic
#   key revocation after the interview is complete. It reads participant details
#   from a JSON file and sends connection instructions via email.
#
# REQUIREMENTS:
#   - jq command must be installed (sudo apt-get install jq)
#   - SSH server must be installed and running
#   - Python 3 for sending emails
#   - JSON file with participant details at /home/USERNAME/college/nueral_networks/Dr_mohsen/project/info.json
#
# JSON FILE FORMAT:
#   {
#     "ssh_public_key": "ssh-rsa AAAA...",
#     "email": "participant@example.com",
#     "name": "Participant Name"
#   }
#
#############################################################

# Display usage if insufficient arguments
if [ $# -lt 3 ]; then
    echo "ERROR: Insufficient arguments"
    echo "USAGE: bash setup_interview.sh USERNAME GMAIL_ADDRESS GMAIL_APP_PASSWORD"
    echo "EXAMPLE: bash setup_interview.sh salma example@gmail.com abcdefghijklmnop"
    exit 1
fi

# Assign arguments to variables
USERNAME="$1"
GMAIL_ADDRESS="$2"
GMAIL_APP_PASSWORD="$3"

# Log file and function
LOG_FILE="/tmp/ssh_key_management.log"
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
    echo "$1"
}

log_message "Starting SSH key management script with username: $USERNAME"

# Configuration variables with dynamic username
BASE_DIR="/home/$USERNAME/college/nueral_networks/Dr_mohsen/project"
SCRIPT_PATH="$BASE_DIR/conductor.py"
VENV_PATH="$BASE_DIR/venv"
SSH_AUTHORIZED_KEYS="/home/$USERNAME/.ssh/authorized_keys"
WORKING_DIR="$BASE_DIR"
JSON_FILE="$BASE_DIR/info.json"
GOOGLE_CREDENTIALS_PATH="$BASE_DIR/interviewai-459420-ddd596af82a8.json"
INTERVIEWER_SCRIPT="$BASE_DIR/interviewer_script.json"
REVOKE_SCRIPT="/home/$USERNAME/revoke_ssh_key.sh"
TEMP_KEY_FILE="/tmp/user_key.pub"
TEMP_PY_SCRIPT="/tmp/send_email.py"

# Check if JSON file exists
if [ ! -f "$JSON_FILE" ]; then
    log_message "Error: JSON file not found at $JSON_FILE"
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    log_message "Error: jq is not installed. Install it with: sudo apt-get install jq"
    exit 1
fi

# Read data from JSON
SSH_PUBLIC_KEY=$(jq -r '.ssh_public_key' "$JSON_FILE")
EMAIL=$(jq -r '.email' "$JSON_FILE")
NAME=$(jq -r '.name' "$JSON_FILE")

# Validation
if [ -z "$SSH_PUBLIC_KEY" ] || [ "$SSH_PUBLIC_KEY" = "null" ]; then
    log_message "Error: 'ssh_public_key' field is missing or empty in $JSON_FILE"
    exit 1
fi
if [ -z "$EMAIL" ] || [ "$EMAIL" = "null" ]; then
    log_message "Error: 'email' field is missing or empty in $JSON_FILE"
    exit 1
fi
if [ -z "$NAME" ] || [ "$NAME" = "null" ]; then
    log_message "Error: 'name' field is missing or empty in $JSON_FILE"
    exit 1
fi

# Check SSH server status
if ! command -v sshd &> /dev/null; then
    log_message "Error: SSH server (sshd) is not installed."
    exit 1
fi
if ! systemctl is-active --quiet ssh; then
    log_message "Starting SSH server..."
    sudo systemctl start ssh
fi

# Set up SSH directory with proper permissions
mkdir -p "/home/$USERNAME/.ssh"
chmod 700 "/home/$USERNAME/.ssh"
chown "$USERNAME:$USERNAME" "/home/$USERNAME/.ssh"

# Set up authorized_keys with proper permissions
touch "$SSH_AUTHORIZED_KEYS"
chmod 600 "$SSH_AUTHORIZED_KEYS"
chown "$USERNAME:$USERNAME" "$SSH_AUTHORIZED_KEYS"

# Verify required files
if [ ! -f "$GOOGLE_CREDENTIALS_PATH" ]; then
    log_message "Error: Google credentials file not found at $GOOGLE_CREDENTIALS_PATH"
    exit 1
fi
if [ ! -f "$INTERVIEWER_SCRIPT" ]; then
    log_message "Error: Interviewer script JSON file not found at $INTERVIEWER_SCRIPT"
    exit 1
fi

# Set permissions
chmod 600 "$GOOGLE_CREDENTIALS_PATH"
chown "$USERNAME:$USERNAME" "$GOOGLE_CREDENTIALS_PATH"
chmod 600 "$INTERVIEWER_SCRIPT"
chown "$USERNAME:$USERNAME" "$INTERVIEWER_SCRIPT"
log_message "Permissions set for sensitive files."

# Create revocation script
log_message "Creating revocation script at $REVOKE_SCRIPT"

cat > "$REVOKE_SCRIPT" << EOF
#!/bin/bash

# This script removes the SSH key from authorized_keys file
# It's designed to be run by the SSH forced command

# Configuration
USERNAME="$USERNAME"
SSH_AUTHORIZED_KEYS="/home/\$USERNAME/.ssh/authorized_keys"
LOG_FILE="$LOG_FILE"
JSON_FILE="$JSON_FILE"
TEMP_FILE="/tmp/authorized_keys.tmp"

# Function to log messages
log_message() {
    echo "\$(date '+%Y-%m-%d %H:%M:%S') - REVOKE: \$1" >> "\$LOG_FILE"
}

log_message "Starting key revocation process"

# Check if the authorized_keys file exists
if [ ! -f "\$SSH_AUTHORIZED_KEYS" ]; then
    log_message "Error: authorized_keys file not found at \$SSH_AUTHORIZED_KEYS"
    exit 1
fi

# Get the key from JSON
if [ -f "\$JSON_FILE" ]; then
    SSH_PUBLIC_KEY=\$(jq -r '.ssh_public_key' "\$JSON_FILE")
    
    if [ -n "\$SSH_PUBLIC_KEY" ] && [ "\$SSH_PUBLIC_KEY" != "null" ]; then
        # Extract the key fingerprint for more reliable matching
        TEMP_KEY_FILE="/tmp/temp_key_file_\$\$.pub"
        echo "\$SSH_PUBLIC_KEY" > "\$TEMP_KEY_FILE"
        KEY_FINGERPRINT=\$(ssh-keygen -lf "\$TEMP_KEY_FILE" 2>/dev/null | awk '{print \$2}')
        rm -f "\$TEMP_KEY_FILE"
        
        if [ -n "\$KEY_FINGERPRINT" ]; then
            log_message "Key fingerprint to remove: \$KEY_FINGERPRINT"
            
            # Strategy 1: Use simple pattern matching
            # Extract a unique part of the key for matching
            KEY_PART=\$(echo "\$SSH_PUBLIC_KEY" | awk '{print \$2}' | cut -c1-20)
            BEFORE_COUNT=\$(grep -c "\$KEY_PART" "\$SSH_AUTHORIZED_KEYS")
            log_message "Found \$BEFORE_COUNT matches for key pattern in authorized_keys"
            
            if [ "\$BEFORE_COUNT" -gt 0 ]; then
                # Make a backup first
                cp "\$SSH_AUTHORIZED_KEYS" "\$SSH_AUTHORIZED_KEYS.bak_\$(date +%s)"
                
                # Remove lines containing the key pattern
                grep -v "\$KEY_PART" "\$SSH_AUTHORIZED_KEYS" > "\$TEMP_FILE"
                cat "\$TEMP_FILE" > "\$SSH_AUTHORIZED_KEYS"
                rm -f "\$TEMP_FILE"
                
                # Verify removal
                AFTER_COUNT=\$(grep -c "\$KEY_PART" "\$SSH_AUTHORIZED_KEYS")
                log_message "After removal: \$AFTER_COUNT matches remain"
                
                if [ "\$AFTER_COUNT" -lt "\$BEFORE_COUNT" ]; then
                    log_message "Key removal successful"
                else
                    log_message "Key removal failed - pattern approach"
                fi
            else
                log_message "Key pattern not found in authorized_keys"
            fi
        else
            log_message "Could not generate key fingerprint"
        fi
    else
        log_message "Invalid or empty SSH key in JSON file"
    fi
else
    log_message "JSON file not found"
fi

# Extra verification by checking if authorized_keys is empty
if [ ! -s "\$SSH_AUTHORIZED_KEYS" ]; then
    log_message "WARNING: authorized_keys file is empty after operation"
fi

log_message "Key revocation process completed"
exit 0
EOF

# Make revocation script executable
chmod +x "$REVOKE_SCRIPT"
chown "$USERNAME:$USERNAME" "$REVOKE_SCRIPT"
log_message "Revocation script created and made executable"

# Write SSH public key to temporary file for validation
echo "$SSH_PUBLIC_KEY" > "$TEMP_KEY_FILE"

# Validate SSH public key format
if ! grep -qE '^(ssh-rsa|ecdsa-sha2-nistp256|ecdsa-sha2-nistp384|ecdsa-sha2-nistp521|ssh-ed25519) [A-Za-z0-9+/=]+' "$TEMP_KEY_FILE"; then
    log_message "Error: Invalid SSH public key format in $JSON_FILE"
    rm -f "$TEMP_KEY_FILE"
    exit 1
fi

# Extract a unique part of the key for the forced command
KEY_PART=$(echo "$SSH_PUBLIC_KEY" | awk '{print $2}' | cut -c1-20)

# Check if key already exists in authorized_keys
if grep -q "$KEY_PART" "$SSH_AUTHORIZED_KEYS"; then
    log_message "Key already exists in authorized_keys, removing it first"
    grep -v "$KEY_PART" "$SSH_AUTHORIZED_KEYS" > "/tmp/authorized_keys.tmp"
    cat "/tmp/authorized_keys.tmp" > "$SSH_AUTHORIZED_KEYS"
    rm -f "/tmp/authorized_keys.tmp"
fi

# Add the public key with script execution forced and automatic revocation
FORCED_COMMAND="command=\"cd $WORKING_DIR && source $VENV_PATH/bin/activate && python $SCRIPT_PATH && bash $REVOKE_SCRIPT\""
echo "$FORCED_COMMAND $SSH_PUBLIC_KEY" >> "$SSH_AUTHORIZED_KEYS"
log_message "Added SSH key with forced command and auto-revocation"

# Check if key was added successfully
if grep -q "$KEY_PART" "$SSH_AUTHORIZED_KEYS"; then
    log_message "SSH key successfully added to authorized_keys"
else
    log_message "ERROR: Failed to add SSH key to authorized_keys"
    exit 1
fi

# Set proper permissions on authorized_keys
chmod 600 "$SSH_AUTHORIZED_KEYS"
chown "$USERNAME:$USERNAME" "$SSH_AUTHORIZED_KEYS"
log_message "Set proper permissions on authorized_keys"

# Clean up temporary key file
rm -f "$TEMP_KEY_FILE"

# Get the machine's IP address
IP_ADDRESS=$(hostname -I | awk '{print $1}' || curl -s ifconfig.me)
if [ -z "$IP_ADDRESS" ]; then
    log_message "Error: Could not determine IP address. Please provide it manually."
    exit 1
fi

# Generate connection instructions
INSTRUCTIONS=$(cat << EOF
To start the interview:
1. Open a terminal on your computer.
2. Use the following command to connect:
   ssh $USERNAME@$IP_ADDRESS
3. The interview script will run automatically.
4. You will be disconnected when the interview is complete.
5. Ensure your SSH private key is set up correctly.

Note: Access will be automatically revoked after the interview completes.
EOF
)

# Create temporary Python script to send email
cat > "$TEMP_PY_SCRIPT" << EOF
import smtplib
from email.mime.text import MIMEText
import os

def send_email(recipient, instructions, name, sender, password):
    """Send an email with connection instructions"""
    try:
        body = f"""
Dear {name},

{instructions}

Best regards,
TechInterviewerAI Team
"""

        msg = MIMEText(body)
        msg['Subject'] = 'TechInterviewerAI: Your Interview Connection Instructions'
        msg['From'] = sender
        msg['To'] = recipient

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.sendmail(msg['From'], msg['To'], msg.as_string())

        print(f"Email sent successfully to {recipient}")
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

success = send_email("$EMAIL", """$INSTRUCTIONS""", "$NAME", "$GMAIL_ADDRESS", "$GMAIL_APP_PASSWORD")
exit(0 if success else 1)
EOF

# Run the Python script to send the email
python3 "$TEMP_PY_SCRIPT"
if [ $? -ne 0 ]; then
    log_message "Error: Failed to send email to $EMAIL"
    rm -f "$TEMP_PY_SCRIPT"
    exit 1
else
    log_message "Email with connection instructions sent successfully to $EMAIL"
fi

# Clean up temporary Python script
rm -f "$TEMP_PY_SCRIPT"

# Output connection instructions to console
echo -e "\n=== Connection Instructions for the Participant ==="
echo "$INSTRUCTIONS"
echo -e "\nInstructions have been emailed to $EMAIL."
echo -e "\nAccess will be automatically revoked after the interview completes."

# Instructions for manual testing/debugging
echo -e "\n===== Testing & Debugging Instructions ====="
echo "1. To manually revoke the SSH key, run: bash $REVOKE_SCRIPT"
echo "2. To check logs, run: cat $LOG_FILE"
echo "3. To see current authorized keys, run: cat $SSH_AUTHORIZED_KEYS"

log_message "SSH key setup completed successfully"
exit 0