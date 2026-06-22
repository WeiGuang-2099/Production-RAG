import { Settings as SettingsIcon } from "lucide-react";
import { useState } from "react";
import { NavLink } from "react-router-dom";
import { SettingsPanel } from "./SettingsPanel";
import { StatusDot } from "./StatusDot";

const linkClass = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-1.5 rounded text-sm ${isActive ? "bg-primary text-white" : "text-ink hover:bg-muted/20"}`;

export function Header() {
  const [open, setOpen] = useState(false);
  return (
    <header className="relative flex items-center justify-between border-b border-muted/30 bg-surface px-6 py-3">
      <div className="flex items-center gap-6">
        <span className="font-semibold text-primary">Production RAG</span>
        <nav className="flex gap-1">
          <NavLink to="/" end className={linkClass}>
            Chat
          </NavLink>
          <NavLink to="/documents" className={linkClass}>
            Documents
          </NavLink>
          <NavLink to="/about" className={linkClass}>
            About
          </NavLink>
        </nav>
      </div>
      <div className="flex items-center gap-4">
        <StatusDot />
        <button aria-label="Settings" className="text-muted hover:text-ink" onClick={() => setOpen((o) => !o)}>
          <SettingsIcon size={18} />
        </button>
        {open && <SettingsPanel onClose={() => setOpen(false)} />}
      </div>
    </header>
  );
}
