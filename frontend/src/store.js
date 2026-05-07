import { create } from "zustand";

export const useAppStore = create((set) => ({
  mode: "Visual", // Visual, Audio, Video
  imageFile: null,
  audioFile: null,
  videoFile: null,
  text: "",
  weight: 0.55,
  result: null,
  loading: false,

  setMode: (mode) => set({ mode, result: null }),
  setImage: (file) => set({ imageFile: file }),
  setAudio: (file) => set({ audioFile: file }),
  setVideo: (file) => set({ videoFile: file }),
  setText: (t) => set({ text: t }),
  setWeight: (w) => set({ weight: w }),
  setResult: (r) => set({ result: r }),
  setLoading: (v) => set({ loading: v }),
  reset: () =>
    set({
      imageFile: null,
      audioFile: null,
      videoFile: null,
      text: "",
      weight: 0.55,
      result: null,
      loading: false,
    }),
}));
