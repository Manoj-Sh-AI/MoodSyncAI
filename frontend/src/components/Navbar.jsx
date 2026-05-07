import React from "react";
import { useAppStore } from "../store.js";

const MODES = ["Visual", "Audio", "Video"];

export default function Navbar() {
  const activeMode = useAppStore((s) => s.mode);
  const setMode = useAppStore((s) => s.setMode);

  return (
    <nav className="flex space-x-2">
      {MODES.map((mode) => (
        <button
          key={mode}
          onClick={() => setMode(mode)}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors
            ${
              activeMode === mode
                ? "bg-sky-500 text-white"
                : "bg-neutral-800 text-neutral-300 hover:bg-neutral-700"
            }`}
        >
          {mode} Input
        </button>
      ))}
    </nav>
  );
}
