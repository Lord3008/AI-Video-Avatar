import { AlertCircle, Loader, Video } from "lucide-react";
import { useState } from "react";
import { signIn, useSession } from "next-auth/react";
import { useRouter } from "next/navigation";

export const NavBar = () => {
  const [loading, setLoading] = useState(false);
  const [avatarLoading, setAvatarLoading] = useState(false);
  const [error, setError] = useState<string>();
  const { data: session, status } = useSession();

  const router = useRouter();

  const HandleCreateAvatar = () => {
    setAvatarLoading(true);
    try {
      if (session && status === "authenticated") {
        router.push("/ai-tutor");
      } else {
        router.push("/login");
      }
    } catch {
    } finally {
      setAvatarLoading(false);
    }
  };

  const handleGoogleSignin = async () => {
    setLoading(true);
    setError("");
    try {
      await signIn("google", {
        callbackUrl: `/ai-tutor`,
      });
      //   toast.success("Successfully logged in!");
    } catch (error) {
      console.error("Error signing in with Google:", error);
      setError("Failed to sign in with Google");
      //   toast.error("Failed to sign in with Google");
    } finally {
      setLoading(false);
    }
  };
  return (
    <nav className="fixed w-full z-50 bg-[#2b1403]/80 backdrop-blur-md border-b border-orange-800/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Video className="h-8 w-8 text-orange-400" />
            <span className="ml-2 text-xl font-bold bg-gradient-to-r from-orange-400 to-amber-400 bg-clip-text text-transparent">
              WiWiAvatars
            </span>
          </div>
          <div className="hidden md:flex items-center space-x-8">
            <a
              href="#"
              className="text-gray-300 hover:text-white transition-colors"
            >
              Home
            </a>
            <a
              href="#"
              className="text-gray-300 hover:text-white transition-colors"
            >
              Features
            </a>
            <a
              href="#"
              className="text-gray-300 hover:text-white transition-colors"
            >
              About
            </a>
            <a
              href="#"
              className="text-gray-300 hover:text-white transition-colors"
            >
              Gallery
            </a>
            <a
              href="#"
              className="text-gray-300 hover:text-white transition-colors"
            >
              Pricing
            </a>
          </div>
          <div className="flex items-center space-x-4">
            <button
              onClick={() => {
                handleGoogleSignin();
              }}
              className="px-4 flex gap-2 py-2 text-sm text-white/80 hover:text-white transition-colors"
            >
              Sign In{loading && <Loader className="animate-spin w-5 h-5" />}
            </button>
            <button
              onClick={() => {
                HandleCreateAvatar();
              }}
              className="px-4 flex gap-2 py-2 text-sm bg-gradient-to-r from-orange-600 to-amber-600 rounded-full hover:opacity-90 transition-opacity"
            >
              Create Avatar
              {avatarLoading && <Loader className="animate-spin w-5 h-5" />}
            </button>
          </div>
         
        {error && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-500/40 rounded-lg flex items-start">
            <AlertCircle className="w-5 h-5 text-red-400 mr-3 mt-0.5 flex-shrink-0" />
            <p className="text-red-300">{error}</p>
          </div>
        )}
        </div>
      </div>
    </nav>
  );
};
