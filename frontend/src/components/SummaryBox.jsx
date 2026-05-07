import React from "react";

export default function SummaryBox({
  mismatch,
  text,
  visual,
  textRes,
  fusion,
}) {
  const accent = mismatch ? "#FF8F00" : "#4f98a3";
  const label = mismatch
    ? "⚠️ Incongruence detected"
    : "✅ Consistent emotional state";
  return (
    <div>
      <div className="text-xs uppercase tracking-widest text-neutral-500 mb-2">
        Generative Summary (Transformer)
      </div>
      <div
        className="rounded-r-xl p-4"
        style={{ background: "#1a1a2e", borderLeft: `4px solid ${accent}` }}
      >
        <div
          className="text-xs font-bold tracking-widest mb-1"
          style={{ color: accent }}
        >
          {label}
        </div>
        <div className="text-sm text-neutral-300 leading-7">{text}</div>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3">
        <Metric
          label="👁 Visual"
          value={
            visual?.top_label?.[0]?.toUpperCase() + visual?.top_label?.slice(1)
          }
          sub={`${visual?.top_score}%`}
        />
        <Metric
          label="💬 Text"
          value={
            textRes?.top_label?.[0]?.toUpperCase() +
            textRes?.top_label?.slice(1)
          }
          sub={`${textRes?.top_score}%`}
        />
        <Metric
          label="🔗 Fused"
          value={
            fusion?.dominant_emotion?.[0]?.toUpperCase() +
            fusion?.dominant_emotion?.slice(1)
          }
          sub={`${fusion?.fused_confidence}%`}
        />
        <Metric
          label="📶 Status"
          value={mismatch ? "Mismatch" : "Aligned"}
          sub=""
        />
      </div>
    </div>
  );
}

function Metric({ label, value, sub }) {
  return (
    <div className="card rounded-lg p-3 text-center">
      <div className="text-xs text-neutral-500">{label}</div>
      <div className="text-base font-semibold">{value}</div>
      <div className="text-xs text-neutral-500">{sub}</div>
    </div>
  );
}
