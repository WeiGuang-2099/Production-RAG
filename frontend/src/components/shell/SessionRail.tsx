import { Info, LibraryBig, Plus, Settings as SettingsIcon, Trash2 } from "lucide-react";
import { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { useChatContext } from "../../context/ChatContext";
import { SettingsPanel } from "../SettingsPanel";
import { StatusDot } from "../StatusDot";

const navClass = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-2 rounded-md px-2 py-1.5 text-sm ${
    isActive ? "bg-highlight text-highlight-ink" : "text-muted hover:bg-surface hover:text-ink"
  }`;

export function SessionRail({ onNavigate }: { onNavigate?: () => void }) {
  const { sessions, activeId, newSession, switchSession, deleteSession } = useChatContext();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const navigate = useNavigate();

  const openSession = (id: string) => {
    switchSession(id);
    navigate("/");
    onNavigate?.();
  };

  return (
    <div className="flex h-full w-full flex-col p-3">
      <div className="px-2 pb-3 pt-1">
        <div className="font-serif text-lg font-semibold text-ink">Production RAG</div>
        <div className="text-xs text-muted">grounded, cited answers</div>
      </div>

      <button
        type="button"
        className="mb-3 flex items-center gap-2 rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink shadow-sm hover:border-primary hover:text-primary"
        onClick={() => {
          newSession();
          navigate("/");
          onNavigate?.();
        }}
      >
        <Plus size={15} /> New chat
      </button>

      <nav className="min-h-0 flex-1 space-y-1 overflow-y-auto" aria-label="Chat sessions">
        {sessions.map((s) => (
          <div
            key={s.id}
            className={`group flex items-center gap-1 rounded-md px-2 py-1.5 text-sm ${
              s.id === activeId ? "bg-highlight text-highlight-ink" : "text-ink hover:bg-surface"
            }`}
          >
            <button
              type="button"
              className="min-w-0 flex-1 truncate text-left"
              title={s.title}
              onClick={() => openSession(s.id)}
            >
              {s.title}
            </button>
            <button
              type="button"
              aria-label={`Delete ${s.title}`}
              className="hidden shrink-0 text-muted hover:text-danger group-hover:block"
              onClick={() => deleteSession(s.id)}
            >
              <Trash2 size={13} />
            </button>
          </div>
        ))}
      </nav>

      <div className="relative mt-3 space-y-1 border-t border-line pt-3">
        <NavLink to="/documents" className={navClass} onClick={onNavigate}>
          <LibraryBig size={15} /> Documents
        </NavLink>
        <NavLink to="/about" className={navClass} onClick={onNavigate}>
          <Info size={15} /> About
        </NavLink>
        <div className="flex items-center justify-between px-2 pt-1">
          <StatusDot />
          <button
            type="button"
            aria-label="Settings"
            className="text-muted hover:text-ink"
            onClick={() => setSettingsOpen((o) => !o)}
          >
            <SettingsIcon size={16} />
          </button>
        </div>
        {settingsOpen && <SettingsPanel onClose={() => setSettingsOpen(false)} />}
      </div>
    </div>
  );
}
