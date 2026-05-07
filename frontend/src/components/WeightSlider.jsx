import React from "react";

export default function WeightSlider({ value, onChange }) {
  return (
    <div className="card rounded-xl p-4">
      <input
        type="range"
        min="0"
        max="1"
        step="0.05"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
      <div className="text-sm text-neutral-400 mt-2">
        <span className="mr-3">👁 Visual: {Math.round(value * 100)}%</span>
        <span>💬 Text: {Math.round((1 - value) * 100)}%</span>
      </div>
    </div>
  );
}
