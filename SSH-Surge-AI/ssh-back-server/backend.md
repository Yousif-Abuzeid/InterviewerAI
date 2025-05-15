
# SSH Key Submission Backend

## Overview

The SSH Key Submission Backend is a FastAPI-based server designed to handle SSH key submissions from a frontend application. It provides an API endpoint to receive user data (username, email, and public SSH key), processes the data, and sends it via email to a designated recipient with a JSON attachment containing the submission details. The backend ensures secure data handling, supports cross-origin requests, and uses asynchronous email processing to maintain performance.

### Key Features

- **API Endpoint**: Exposes a `/api/send-ssh-key/` POST endpoint to accept and process SSH key submissions.
- **Data Validation**: Uses Pydantic models to validate incoming data, ensuring username, email, and SSH key are provided correctly.
- **Email Notifications**: Sends an HTML email with submission details and a JSON attachment to a specified recipient, processed in the background to avoid blocking the API response.
- **CORS Support**: Configured to allow cross-origin requests from multiple frontend origins, including local development and production environments.
- **Temporary File Handling**: Creates a temporary JSON file for email attachments, ensuring cleanup after sending.
- **Asynchronous Processing**: Leverages FastAPI’s `BackgroundTasks` for non-blocking email delivery.

## Getting Started

Follow these steps to set up and run the backend locally.

### Prerequisites

- **Python**: Requires Python 3.8 or higher.
- **pip**: Ensure pip is installed for managing Python dependencies.
- **Virtual Environment** (recommended): Use `venv` or another tool to isolate dependencies.

### Installation

1. **Clone the Repository**

   ```bash
   git clone <YOUR_GIT_URL>
   ```

2. **Navigate to the Project Directory**

   ```bash
   cd ssh-key-submission-backend
   ```

3. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install Dependencies**

   ```bash
   pip install fastapi uvicorn fastapi-mail aiofiles pydantic python-dotenv
   ```

5. **Set Up Environment Variables**

   Create a `.env` file in the project root with the following email configuration:

   ```env
   MAIL_USERNAME=your-email@example.com
   MAIL_PASSWORD=your-email-password
   MAIL_FROM=your-email@example.com
   MAIL_PORT=587
   MAIL_SERVER=smtp.example.com
   MAIL_STARTTLS=True
   MAIL_SSL_TLS=False
   ```

   Replace the values with your email provider’s SMTP settings (e.g., Gmail, SendGrid).

6. **Start the Development Server**

   ```bash
   uvicorn main:app --reload
   ```

   This launches the FastAPI server with auto-reloading. The API will be accessible at `http://localhost:8000`. The endpoint documentation is available at `http://localhost:8000/docs`.

## Technologies Used

The backend is built with the following technologies:

- **FastAPI**: A modern, high-performance web framework for building APIs with Python.
- **Pydantic**: Provides data validation and serialization for incoming requests.
- **fastapi-mail**: Handles email sending with SMTP configuration and attachment support.
- **aiofiles**: Enables asynchronous file operations for creating temporary JSON attachments.
- **python-dotenv**: Loads environment variables from a `.env` file for secure configuration.
- **Uvicorn**: An ASGI server for running the FastAPI application.
- **CORS Middleware**: Configures cross-origin resource sharing to support frontend integration.

## Project Structure

- `main.py`: The main application file containing the FastAPI setup, CORS configuration, email logic, and API endpoint.
- `.env`: Stores environment variables for email configuration (not tracked in version control).
- `requirements.txt` (optional): Lists dependencies for easy installation.
- `.gitignore`: Excludes sensitive files (e.g., `.env`, virtual environment) from version control.

## How It Works

The backend provides a single POST endpoint, `/api/send-ssh-key/`, which:

1. **Receives Data**: Accepts a JSON payload with `username`, `email`, and `sshKey`, validated using a Pydantic `SshKeyData` model.
2. **Processes Submission**:
   - Converts the data to a dictionary and serializes it as formatted JSON.
   - Schedules an email task using `BackgroundTasks` to avoid blocking the API response.
3. **Sends Email**:
   - Creates a temporary JSON file containing the submission data.
   - Constructs an HTML email with a summary of the submission (username and email) and attaches the JSON file.
   - Sends the email to a hardcoded recipient (`salmamohammedhamed2@gmail.com`) using `fastapi-mail` with SMTP configuration.
   - Cleans up the temporary file after sending.
4. **Returns Response**: Immediately responds with a success message, indicating that the data is being processed.

The CORS middleware allows requests from specified origins, including local development servers (`http://localhost:5173`) and the deployed frontend (`https://ssh-key-form-builder.onrender.com`), ensuring seamless integration with the frontend application.

