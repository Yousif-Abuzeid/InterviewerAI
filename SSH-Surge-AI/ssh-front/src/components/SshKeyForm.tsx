
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";

const SshKeyForm = () => {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [sshKey, setSshKey] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Update this URL to your deployed backend on Render
  const backendApiUrl = "https://ssh-backend-server.onrender.com/api/send-ssh-key/";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!username.trim()) {
      toast.error("Username is required");
      return;
    }

    if (!email.trim() || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      toast.error("Please enter a valid email address");
      return;
    }

    if (!sshKey.trim() || !sshKey.startsWith("ssh-")) {
      toast.error("Please enter a valid SSH public key");
      return;
    }

    setIsSubmitting(true);

    const formData = {
      username,
      email,
      jobDescription,
      sshKey,
    };

    const response = await fetch(backendApiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    });

    if (response.ok) {
      const result = await response.json();
      toast.success(result.message || "SSH Key submitted successfully and email process initiated!");
      setUsername("");
      setEmail("");
      setJobDescription("");
      setSshKey("");
    } else {
      let errorMessage = "Failed to submit SSH key.";
      try {
        const errorResult = await response.json();
        errorMessage = errorResult.detail || errorResult.message || errorMessage;
      } catch (parseError) {
        // If parsing error response fails, use the generic message
      }
      toast.error(errorMessage);
      console.error("Error submitting SSH key:", response.statusText, await response.text());
    }
    setIsSubmitting(false);
  };

  return (
    <div className="w-full max-w-md mx-auto rounded-lg border border-gray-200 shadow-sm bg-white p-6">
      <h2 className="text-2xl font-semibold mb-6 text-gray-900">
        Add SSH Key
      </h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="username" className="text-sm font-medium">
            Username
          </Label>
          <Input
            id="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="johndoe"
            className="w-full"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="email" className="text-sm font-medium">
            Email
          </Label>
          <Input
            id="email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="john@example.com"
            className="w-full"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="job-description" className="text-sm font-medium">
            Job Description
          </Label>
          <Textarea
            id="job-description"
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            placeholder="Please describe your current role or position"
            className="w-full resize-y h-24"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="ssh-key" className="text-sm font-medium">
            Public SSH Key
          </Label>
          <Textarea
            id="ssh-key"
            value={sshKey}
            onChange={(e) => setSshKey(e.target.value)}
            placeholder="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC..."
            className="font-mono text-sm h-32 resize-y"
          />
          <p className="text-xs text-gray-500">
            Paste your public SSH key. It typically starts with
            "ssh-rsa", "ssh-ed25519", etc.
          </p>
        </div>

        <Button
          type="submit"
          className="w-full bg-blue-600 hover:bg-blue-700"
          disabled={isSubmitting}
        >
          {isSubmitting ? "Submitting..." : "Submit SSH Key"}
        </Button>
      </form>
    </div>
  );
};

export default SshKeyForm;
