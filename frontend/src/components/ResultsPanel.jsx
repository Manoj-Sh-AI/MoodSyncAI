import React from "react";

const COLORS = {
  happy: "#4CAF50",
  joy: "#4CAF50",
  positive: "#4CAF50",
  sad: "#5C6BC0",
  sadness: "#5C6BC0",
  angry: "#EF5350",
  anger: "#EF5350",
  negative: "#EF5350",
  fear: "#AB47BC",
  disgust: "#FF7043",
  surprise: "#FFA726",
  neutral: "#78909C",
};

function Bars({ scores }) {
  const items = Object.entries(scores || {}).sort((a, b) => b[1] - a[1]);
  return (
    <div>
      {items.map(([label, score]) => (
        <div key={label} className="mb-2">
          <div className="flex justify-between text-xs text-neutral-400 mb-1">
            <span>{label[0].toUpperCase() + label.slice(1)}</span>
            <span>{score}%</span>
          </div>
          <div className="h-2 rounded bg-neutral-800">
            <div
              className="h-2 rounded"
              style={{
                width: `${score}%`,
                background: COLORS[label] || "#90A4AE",
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function ResultsPanel({ visual, audio, text, fusion }) {
  return (
    <div className="grid md:grid-cols-3 gap-4">
      {(visual || audio) && (
        <div className="card rounded-xl p-4">
          <div className="text-xs uppercase tracking-widest text-neutral-500">
            {visual ? "Visual Emotion (CNN/ViT)" : "Audio Emotion"}
          </div>
          <h3
            className="text-lg font-semibold mt-1"
            style={{ color: COLORS[(visual || audio)?.top_label] || "#e0e0e0" }}
          >
            {visual ? "👁" : "🎤"}{" "}
            {((visual || audio)?.top_label || "")
              .toString()
              .replace(/^./, (c) => c.toUpperCase())}
          </h3>
          <div className="text-xs text-neutral-400">
            Confidence: {(visual || audio)?.top_score}%
          </div>
          <div className="mt-3">
            <Bars scores={(visual || audio)?.all_scores} />
          </div>
        </div>
      )}

      <div className="card rounded-xl p-4">
        <div className="text-xs uppercase tracking-widest text-neutral-500">
          Textual Polarity
        </div>
        <h3
          className="text-lg font-semibold mt-1"
          style={{ color: COLORS[text?.top_label] || "#e0e0e0" }}
        >
          💬 {text?.top_label?.[0]?.toUpperCase() + text?.top_label?.slice(1)}
        </h3>
        <div className="text-xs text-neutral-400">
          Confidence: {text?.top_score}%
        </div>
        <div className="mt-3">
          <Bars scores={text?.all_scores} />
        </div>
      </div>

      <div className="card rounded-xl p-4">
        <div className="text-xs uppercase tracking-widest text-neutral-500">
          Fusion Result
        </div>
        <div className="mt-2">
          {fusion?.mismatch ? (
            <div className="inline-block bg-amber-600 text-white text-sm font-semibold px-3 py-1 rounded-full">
              ⚠️ MISMATCH DETECTED
            </div>
          ) : (
            <div className="inline-block bg-green-700 text-white text-sm font-semibold px-3 py-1 rounded-full">
              ✅ SIGNALS ALIGNED
            </div>
          )}
        </div>
        <div className="mt-3 text-sm">
          <span className="inline-block card rounded-md px-3 py-1 mr-2 mb-2">
            👁{" "}
            {visual?.top_label?.[0]?.toUpperCase() +
              visual?.top_label?.slice(1)}{" "}
            — <em>{fusion?.visual_polarity}</em>
          </span>
          <span className="inline-block card rounded-md px-3 py-1 mr-2 mb-2">
            💬 {text?.top_label?.[0]?.toUpperCase() + text?.top_label?.slice(1)}{" "}
            — <em>{fusion?.text_polarity}</em>
          </span>
          <span className="inline-block card rounded-md px-3 py-1 mr-2 mb-2">
            🔗 Fused confidence: {fusion?.fused_confidence}%
          </span>
          <span className="inline-block card rounded-md px-3 py-1 mr-2 mb-2">
            🎯 Dominant:{" "}
            {fusion?.dominant_emotion?.[0]?.toUpperCase() +
              fusion?.dominant_emotion?.slice(1)}
          </span>
        </div>
      </div>
    </div>
  );
}
