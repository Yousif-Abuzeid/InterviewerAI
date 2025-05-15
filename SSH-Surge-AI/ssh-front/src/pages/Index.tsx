
import SshKeyForm from "@/components/SshKeyForm";

const Index = () => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 p-4">
      <div className="w-full max-w-md mb-8 text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Surge AI Interviewer</h1>
        <p className="text-gray-600">
          Add your SSH key to access our secure services
        </p>
      </div>
      
      <SshKeyForm />
      
      <div className="mt-8 text-center text-sm text-gray-500">
        <p className="mb-2">
          To find your public SSH key follow the instructions in our{" "}
          <a 
            href="https://drive.google.com/file/d/1SeBZPYXfsBVRqh-8AqioAZ4fdYe_2hzn/view?usp=drivesdk" 
            target="_blank" 
            rel="noopener noreferrer" 
            className="text-blue-600 hover:text-blue-800 underline"
          >
            PDF guide
          </a>
        </p>
        <p>
          Your SSH keys are securely stored and only used for authentication purposes.
        </p>
      </div>
    </div>
  );
};

export default Index;
