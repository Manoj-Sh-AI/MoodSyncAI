import React from "react";
import Navbar from "./components/Navbar.jsx";
import ImageUploader from "./components/ImageUploader.jsx";
import AudioUploader from "./components/AudioUploader.jsx";
import VideoUploader from "./components/VideoUploader.jsx";
import TextInput from "./components/TextInput.jsx";
import WeightSlider from "./components/WeightSlider.jsx";
import ResultsPanel from "./components/ResultsPanel.jsx";
import SummaryBox from "./components/SummaryBox.jsx";
import DebugPanel from "./components/DebugPanel.jsx";
import { analyzeImage, analyzeAudio, analyzeVideo } from "./api.js";
import { useAppStore } from "./store.js";

export default function App() {
  const {
    mode,
    imageFile,
    audioFile,
    videoFile,
    text,
    weight,
    loading,
    result,
    setImage,
    setAudio,
    setVideo,
    setText,
    setWeight,
    setLoading,
    setResult,
  } = useAppStore();

  const canAnalyze =
    (mode === "Visual" && imageFile && text.trim().length > 0) ||
    (mode === "Audio" && audioFile && text.trim().length > 0) ||
    (mode === "Video" && videoFile && text.trim().length > 0);

  const onAnalyze = async () => {
    if (!canAnalyze) return;
    setLoading(true);
    try {
      let data;
      if (mode === "Visual") {
        data = await analyzeImage({
          image: imageFile,
          text,
          visual_weight: weight,
        });
      } else if (mode === "Audio") {
        data = await analyzeAudio({
          audio: audioFile,
          text,
        });
      } else if (mode === "Video") {
        data = await analyzeVideo({
          video: videoFile,
          text,
          visual_weight: weight,
        });
      }
      setResult(data);
    } catch (e) {
      alert(e.message || "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-5 md:p-8 max-w-6xl mx-auto">
      <header className="mb-6">
        <h1 className="text-3xl font-extrabold">🧠 MoodSyncAI</h1>
        <p className="text-sm text-neutral-400">
          Multi-Modal Sentiment & Emotion Analyzer · CNN + Transformer + Fusion
        </p>
        <hr className="border-neutral-800 mt-3" />
      </header>

      <div className="mb-4">
        <Navbar />
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div>
          <div className="text-xs uppercase tracking-widest text-neutral-500 mb-2">
            {mode} Input — {mode === "Visual" ? "Face Image" : "Content"}
          </div>
          {mode === "Visual" && <ImageUploader onFile={setImage} />}
          {mode === "Audio" && <AudioUploader onFile={setAudio} />}
          {mode === "Video" && <VideoUploader />}
        </div>
        <div>
          <div className="text-xs uppercase tracking-widest text-neutral-500 mb-2">
            Text Input — What They Said
          </div>
          <TextInput value={text} onChange={setText} />
          {mode === "Visual" && (
            <>
              <div className="text-xs uppercase tracking-widest text-neutral-500 mt-4 mb-2">
                Fusion Weight
              </div>
              <WeightSlider value={weight} onChange={setWeight} />
            </>
          )}
        </div>
      </div>

      <div className="mt-5">
        <button
          className="button-primary w-full disabled:opacity-40"
          onClick={onAnalyze}
          disabled={!canAnalyze || loading}
        >
          {loading ? "Running analysis…" : "🔍 Analyze Emotional State"}
        </button>
        {!canAnalyze && mode === "Visual" && (
          <p className="text-sm text-neutral-500 mt-2">
            ⬆️ Upload a face image and enter text to enable analysis.
          </p>
        )}
        {/* Mode-specific helpers can be added here if needed */}
      </div>

      {result && (
        <section className="mt-6">
          <h2 className="text-xl font-semibold mb-3">📊 Results</h2>
          <ResultsPanel
            visual={result.visual}
            audio={result.audio}
            text={result.text}
            fusion={result.fusion}
          />
          <div className="mt-4">
            <SummaryBox
              mismatch={result.fusion?.mismatch}
              text={result.summary}
              visual={result.visual || result.audio}
              textRes={result.text}
              fusion={result.fusion}
            />
          </div>
          <div className="mt-4">
            <DebugPanel data={result} />
          </div>
        </section>
      )}

      <footer className="text-xs text-neutral-600 mt-10">© MoodSyncAI</footer>
    </div>
  );
}
