import { Route, Routes } from "react-router-dom";
import { AppShell } from "./components/shell/AppShell";
import { ChatProvider } from "./context/ChatContext";
import { AboutPage } from "./pages/AboutPage";
import { ChatPage } from "./pages/ChatPage";
import { DocumentsPage } from "./pages/DocumentsPage";

export default function App() {
  return (
    <ChatProvider>
      <AppShell>
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/documents" element={<DocumentsPage />} />
          <Route path="/about" element={<AboutPage />} />
        </Routes>
      </AppShell>
    </ChatProvider>
  );
}
