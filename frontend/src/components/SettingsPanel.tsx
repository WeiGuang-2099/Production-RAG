import { motion } from "framer-motion";
import { useSettings } from "../context/SettingsContext";

export function SettingsPanel({ onClose }: { onClose: () => void }) {
  const { client, setBaseUrl, setApiKey } = useSettings();
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="absolute bottom-full left-0 z-50 mb-2 w-72 rounded-lg border border-line bg-surface p-4 shadow-xl"
    >
      <label className="block text-xs font-medium text-muted">API URL</label>
      <input
        className="mt-1 w-full rounded border border-muted/50 px-2 py-1 text-sm"
        value={client.baseUrl}
        onChange={(e) => setBaseUrl(e.target.value)}
      />
      <label className="mt-3 block text-xs font-medium text-muted">API key (optional)</label>
      <input
        type="password"
        className="mt-1 w-full rounded border border-muted/50 px-2 py-1 text-sm"
        value={client.apiKey ?? ""}
        onChange={(e) => setApiKey(e.target.value)}
      />
      <button className="mt-4 w-full rounded bg-primary px-3 py-1.5 text-sm text-white hover:bg-primary-hover" onClick={onClose}>
        Done
      </button>
    </motion.div>
  );
}
