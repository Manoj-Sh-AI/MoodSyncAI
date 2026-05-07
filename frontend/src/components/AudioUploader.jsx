import React, { useRef, useState } from "react";
import { useAppStore } from "../store.js";

export default function AudioUploader() {
  const inputRef = useRef();
  const audioFile = useAppStore((s) => s.audioFile);
  const setAudio = useAppStore((s) => s.setAudio);
  const [meta, setMeta] = useState(null);

  const onPick = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setAudio(file);
    setMeta(`${file.type} · ${(file.size / 1024).toFixed(0)} KB`);
  };

  return (
    <div className="card rounded-xl p-4">
      <input
        ref={inputRef}
        type="file"
        accept="audio/mpeg,audio/wav,audio/ogg"
        className="hidden"
        onChange={onPick}
      />
      <div className="border-2 border-dashed border-neutral-800 rounded-xl p-8 text-center">
        {!audioFile ? (
          <>
            <div className="text-4xl mb-2">🎤</div>
            <div className="text-neutral-400">Upload a voice recording</div>
            <div className="text-xs text-neutral-600 mt-1">MP3 · WAV · OGG</div>
            <button
              className="button-primary mt-4"
              onClick={() => inputRef.current?.click()}
            >
              Choose Audio File
            </button>
          </>
        ) : (
          <>
            <div className="text-4xl mb-2">✅</div>
            <div className="text-neutral-400">
              Audio file selected:
              <br />
              <span className="font-mono text-sm">{audioFile.name}</span>
            </div>
            <div className="text-xs text-neutral-500 mt-2">{meta}</div>
          </>
        )}
      </div>
    </div>
  );
}
