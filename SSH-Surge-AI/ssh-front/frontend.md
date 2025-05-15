
# SSH Key Submission Application

## Overview

The SSH Key Submission Application is a web-based tool designed to securely collect and submit public SSH keys, usernames, email addresses, and job descriptions from users. The application features a responsive, user-friendly form with robust client-side validation, real-time feedback via toast notifications, and seamless integration with a backend API for data processing. The primary purpose is to facilitate secure SSH key submission for authentication purposes, ensuring a smooth and reliable user experience.

### Key Features

- **Interactive Form**: Allows users to input their username, email, job description, and public SSH key through an intuitive interface.
- **Input Validation**: Enforces rules such as non-empty usernames, valid email formats, and SSH keys starting with recognized prefixes (e.g., "ssh-rsa" or "ssh-ed25519").
- **Backend Communication**: Submits form data to a backend API hosted on Render using a POST request, with proper error handling.
- **User Feedback**: Displays toast notifications for successful submissions, validation errors, or server-side issues, enhancing usability.
- **Responsive Design**: Built with Tailwind CSS and shadcn-ui, the interface is accessible and adapts seamlessly to various screen sizes.
- **State Management**: Leverages React’s `useState` hook for efficient form state handling and submission status tracking.

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

- **Node.js & npm**: Requires Node.js (version 18 or higher) and npm. Install them using [nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

### Installation

1. **Clone the Repository**

   ```bash
   git clone <YOUR_GIT_URL>
   ```

2. **Navigate to the Project Directory**

   ```bash
   cd ssh-key-submission
   ```

3. **Install Dependencies**

   ```bash
   npm install
   ```

4. **Start the Development Server**

   ```bash
   npm run dev
   ```

   This launches the application with auto-reloading. Open `http://localhost:5173` in your browser to access the app.

## Technologies Used

The project is built with the following technologies:

- **Vite**: A high-performance build tool and development server for modern web applications.
- **TypeScript**: Ensures type safety and enhances code maintainability.
- **React**: Powers the dynamic and interactive user interface.
- **shadcn-ui**: Provides accessible, customizable UI components for a polished look.
- **Tailwind CSS**: Enables rapid styling with a utility-first approach.
- **Sonner**: Delivers toast notifications for real-time user feedback.
- **Fetch API**: Facilitates HTTP requests to the backend server.

## Project Structure

- `src/`: Core source code directory.
  - `components/`: Contains the `SshKeyForm` component and other reusable components.
  - `components/ui/`: Houses shadcn-ui components (`Button`, `Input`, `Label`, `Textarea`).
  - `App.tsx`: Main application component rendering the form.
- `public/`: Stores static assets like images or icons.
- `package.json`: Specifies project metadata, scripts, and dependencies.
- `.gitignore`: Excludes irrelevant files from version control.

## How It Works

The application revolves around the `SshKeyForm` component, which provides the following functionality:

1. **Form State Management**: Uses React’s `useState` hook to manage input fields (username, email, job description, SSH key) and submission status.
2. **Client-Side Validation**:
   - Ensures the username is not empty.
   - Validates email format using a regular expression.
   - Checks that the SSH key starts with a valid prefix (e.g., "ssh-rsa", "ssh-ed25519").
3. **Backend Integration**: Submits form data as JSON to the backend API at `https://ssh-backend-server.onrender.com/api/send-ssh-key/` via a POST request.
4. **Response Handling**:
   - On success, displays a toast notification and resets the form.
   - On failure, shows an error message parsed from the server response or a generic fallback.
5. **User Interface**: Features a centered, card-like form with clear labels, placeholders, and a note explaining SSH key formats. The design is clean and professional, using Tailwind CSS and shadcn-ui components.

The application prioritizes usability, security, and feedback, making it an effective tool for SSH key submission.

