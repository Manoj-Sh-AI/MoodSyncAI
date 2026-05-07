import React, { useRef, useState } from "react";
import { useAppStore } from "../store.js";

export default function VideoUploader() {
  const inputRef = useRef();
  const videoFile = useAppStore((s) => s.videoFile);
  const setVideo = useAppStore((s) => s.setVideo);
  const [preview, setPreview] = useState(null);
  const [meta, setMeta] = useState(null);

  const onPick = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setVideo(file);
    const url = URL.createObjectURL(file);
    setPreview(url);
    setMeta(`${file.type} · ${(file.size / 1024).toFixed(0)} KB`);
  };

  return (
    <div className="card rounded-xl p-4">
      <input
        ref={inputRef}
        type="file"
        accept="video/mp4,video/webm,video/quicktime"
        className="hidden"
        onChange={onPick}
      />
      <div className="border-2 border-dashed border-neutral-800 rounded-xl p-8 text-center">
        {!preview ? (
          <>
            <div className="text-4xl mb-2">📹</div>
            <div className="text-neutral-400">Upload a video of a face</div>
            <div className="text-xs text-neutral-600 mt-1">
              MP4 · WEBM · MOV
            </div>
            <button
              className="button-primary mt-4"
              onClick={() => inputRef.current?.click()}
            >
              Choose Video File
            </button>
          </>
        ) : (
          <>
            <video src={preview} className="w-full rounded-lg" controls />
            <div className="text-xs text-neutral-500 mt-2">{meta}</div>
          </>
        )}
      </div>
    </div>
  );
}
