import { Route, Routes } from "react-router-dom";
import { Header } from "./components/Header";
import { ChatProvider } from "./context/ChatContext";
import { AboutPage } from "./pages/AboutPage";
import { ChatPage } from "./pages/ChatPage";
import { DocumentsPage } from "./pages/DocumentsPage";

export default function App() {
  return (
    <div className="min-h-screen">
      <Header />
      <ChatProvider>
        <main className="mx-auto max-w-5xl px-4 py-6">
          <Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/documents" element={<DocumentsPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
      </ChatProvider>
    </div>
  );
}
