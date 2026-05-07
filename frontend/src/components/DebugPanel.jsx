import React from "react";

export default function DebugPanel({ data }) {
  const pretty = JSON.stringify(data, null, 2);
  return (
    <details className="card rounded-xl p-4">
      <summary className="cursor-pointer select-none">
        🔬 Raw model output (debug)
      </summary>
      <pre className="text-xs overflow-auto mt-2 bg-neutral-900 p-3 rounded border border-neutral-800">
        <code>{pretty}</code>
      </pre>
    </details>
  );
}
