import imaplib
import email
from email.header import decode_header
import time
import os
import logging
import subprocess
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("email_downloader.log"),
        logging.StreamHandler()
    ]
)

# Configuration
EMAIL = "salmamohammedhamed2@gmail.com"  # Your email address
APP_PASSWORD = "avpztrbqblmassxy"  # App password (not your regular password)
SUBJECT_KEYWORD = "SSH Key Submission from"  # Subject keyword to search for
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Base directory (current script location)
JSON_FILE = os.path.join(BASE_DIR, "info.json")  # Fixed path for the JSON file
CHECK_INTERVAL = 60  # Check interval in seconds

# Post-processing configuration
RUN_INTERVIEWER = True  # Whether to run interviewer.py after download
INTERVIEWER_PATH = "interviewer.py"  # Path to interviewer.py
RUN_SSH_SETUP = True  # Whether to run the SSH setup script
SSH_SETUP_PATH = "./setup_ssh_access.sh"  # Path to the SSH setup script
USERNAME = "salma"  # Username for SSH setup

def check_email():
    """Connect to Gmail and check for emails with the specified subject line"""
    try:
        logging.info(f"Checking for emails with subject containing '{SUBJECT_KEYWORD}'...")
        
        # Connect to Gmail IMAP server
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(EMAIL, APP_PASSWORD)
        mail.select("inbox")
        
        # Search for emails containing the subject keyword
        status, messages = mail.search(None, f'(SUBJECT "{SUBJECT_KEYWORD}" UNSEEN)')
        
        if not messages[0]:
            logging.info("No new matching emails found.")
            mail.logout()
            return
            
        message_ids = messages[0].split()
        logging.info(f"Found {len(message_ids)} new email(s) matching the criteria.")
        
        for num in message_ids:
            try:
                # Fetch the email
                status, msg_data = mail.fetch(num, "(RFC822)")
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)
                
                # Get and decode the subject
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding or "utf-8")
                
                # Extract sender name from subject (after "SSH Key Submission from ")
                try:
                    sender_name = subject.replace(SUBJECT_KEYWORD, "").strip()
                    logging.info(f"Processing email from: {sender_name}")
                except:
                    sender_name = "unknown"
                    logging.warning("Could not extract sender name from subject")
                
                # Process attachments
                found_json = False
                for part in msg.walk():
                    content_disposition = part.get("Content-Disposition", "")
                    if "attachment" in content_disposition:
                        filename = part.get_filename()
                        if filename and filename.lower().endswith(".json"):
                            # Save directly to the fixed JSON file path
                            payload = part.get_payload(decode=True)
                            with open(JSON_FILE, "wb") as f:
                                f.write(payload)
                            logging.info(f"Saved JSON attachment as {JSON_FILE}")
                            found_json = True
                            break
                
                if not found_json:
                    logging.warning(f"No JSON attachment found in email with subject: {subject}")
                    mail.store(num, '+FLAGS', r'(\Seen)')  # Mark as read even if no JSON found
                    continue  # Skip to next email
                    
                # Run post-processing scripts if requested
                if found_json and RUN_INTERVIEWER:
                    try:
                        # Parse the JSON to extract jobDescription
                        import json
                        with open(JSON_FILE, 'r') as f:
                            try:
                                json_data = json.load(f)
                                extracted_username = json_data.get('username', '')
                                extracted_email = json_data.get('email', '')
                                job_description = json_data.get('jobDescription', '')
                                ssh_key = json_data.get('sshKey', '')
                                
                                logging.info(f"Extracted data from JSON: username={extracted_username}, email={extracted_email}")
                                logging.info(f"Found job description of length: {len(job_description)}")
                            except json.JSONDecodeError as je:
                                logging.error(f"Failed to parse JSON file: {str(je)}")
                                job_description = ""
                        
                        # Run interviewer.py with the job description as argument
                        logging.info(f"Running interviewer.py with extracted job description")
                        interviewer_result = subprocess.run(
                            ["python", INTERVIEWER_PATH, "--job-description", job_description],
                            check=True,
                            capture_output=True,
                            text=True
                        )
                        logging.info(f"interviewer.py completed with output: {interviewer_result.stdout}")
                        
                        # Run SSH setup script after interviewer completes
                        if RUN_SSH_SETUP:
                            logging.info(f"Running SSH setup script with username from JSON: {extracted_username}")
                            ssh_result = subprocess.run(
                                [SSH_SETUP_PATH, USERNAME, EMAIL, APP_PASSWORD],
                                check=True,
                                capture_output=True,
                                text=True
                            )
                            logging.info(f"SSH setup completed with output: {ssh_result.stdout}")
                    except subprocess.CalledProcessError as e:
                        logging.error(f"Error running post-processing scripts: {str(e)}")
                        logging.error(f"Error output: {e.stderr}")
                    except Exception as e:
                        logging.error(f"Unexpected error during post-processing: {str(e)}")
                
                # Mark as read
                mail.store(num, '+FLAGS', r'(\Seen)')
                
            except Exception as e:
                logging.error(f"Error processing email: {str(e)}")
        
        mail.logout()
        
    except Exception as e:
        logging.error(f"Error checking email: {str(e)}")
        
        mail.logout()
        
    except Exception as e:
        logging.error(f"Error checking email: {str(e)}")

def main():
    """Main function to run the email checker periodically"""
    logging.info("Starting email monitoring service")
    try:
        while True:
            check_email()
            logging.debug(f"Sleeping for {CHECK_INTERVAL} seconds...")
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        logging.info("Program terminated by user")
    except Exception as e:
        logging.critical(f"Critical error: {str(e)}")

if __name__ == "__main__":
    main()