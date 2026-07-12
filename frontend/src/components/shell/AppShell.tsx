import { Menu, X } from "lucide-react";
import { useState, type ReactNode } from "react";
import { SessionRail } from "./SessionRail";

export function AppShell({ children }: { children: ReactNode }) {
  const [drawerOpen, setDrawerOpen] = useState(false);
  return (
    <div className="flex h-screen bg-bg">
      <aside className="hidden w-[260px] shrink-0 border-r border-line bg-sunken lg:block">
        <SessionRail />
      </aside>

      {/* <lg: off-canvas drawer */}
      <button
        type="button"
        aria-label="Open menu"
        className="fixed left-3 top-3 z-30 rounded-md border border-line bg-surface p-2 text-ink shadow-sm lg:hidden"
        onClick={() => setDrawerOpen(true)}
      >
        <Menu size={16} />
      </button>
      {drawerOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-ink/30" onClick={() => setDrawerOpen(false)} />
          <div className="absolute inset-y-0 left-0 w-[280px] border-r border-line bg-sunken shadow-xl">
            <button
              type="button"
              aria-label="Close menu"
              className="absolute right-3 top-3 text-muted hover:text-ink"
              onClick={() => setDrawerOpen(false)}
            >
              <X size={16} />
            </button>
            <SessionRail onNavigate={() => setDrawerOpen(false)} />
          </div>
        </div>
      )}

      <main className="min-w-0 flex-1 overflow-hidden">{children}</main>
    </div>
  );
}
