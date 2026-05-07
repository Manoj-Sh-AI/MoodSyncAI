import React, { useRef, useState } from "react";

export default function ImageUploader({ onFile }) {
  const inputRef = useRef();
  const [preview, setPreview] = useState(null);
  const [meta, setMeta] = useState(null);

  const onPick = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    onFile(file);
    const url = URL.createObjectURL(file);
    setPreview(url);
    setMeta(`${file.type} · ${(file.size / 1024).toFixed(0)} KB`);
  };

  return (
    <div className="card rounded-xl p-4">
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        className="hidden"
        onChange={onPick}
      />
      <div className="border-2 border-dashed border-neutral-800 rounded-xl p-8 text-center">
        {!preview ? (
          <>
            <div className="text-4xl mb-2">📷</div>
            <div className="text-neutral-400">Upload a face image to begin</div>
            <div className="text-xs text-neutral-600 mt-1">
              JPG · PNG · WEBP
            </div>
            <button
              className="button-primary mt-4"
              onClick={() => inputRef.current?.click()}
            >
              Choose Image
            </button>
          </>
        ) : (
          <>
            <img
              src={preview}
              className="w-full rounded-lg"
              alt="Uploaded face"
            />
            <div className="text-xs text-neutral-500 mt-2">{meta}</div>
          </>
        )}
      </div>
    </div>
  );
}
