import React from "react";

const EXAMPLES = [
  "No, I think the project is going really well.",
  "Everything is fine, don't worry about me.",
  "I am so excited about this opportunity!",
  "I'm not sure I can handle this right now.",
];

export default function TextInput({ value, onChange }) {
  return (
    <div className="card rounded-xl p-4">
      <div className="text-xs text-neutral-500 mb-1">Try an example →</div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-2">
        {EXAMPLES.map((e) => (
          <button
            key={e}
            className="text-xs card rounded-md p-2 hover:bg-neutral-800 transition-colors"
            onClick={() => onChange(e)}
          >
            {e.length > 22 ? e.slice(0, 22) + "…" : e}
          </button>
        ))}
      </div>
      <textarea
        className="w-full bg-neutral-900 rounded-lg border border-neutral-800 p-3 text-sm outline-none focus:ring-2 focus:ring-sky-600 min-h-[110px]"
        placeholder="e.g. No, I think the project is going really well."
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
      <div className="text-right text-xs text-neutral-500 mt-1">
        {value.trim()
          ? `${value.split(/\s+/).length} words · ${value.length} chars`
          : "0 words · 0 chars"}
      </div>
    </div>
  );
}
