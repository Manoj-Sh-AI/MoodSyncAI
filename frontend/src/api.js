import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8080";

export async function analyzeImage({ image, text, visual_weight }) {
  const form = new FormData();
  form.append("text", text);
  form.append("visual_weight", String(visual_weight));
  form.append("image", image);
  try {
    const res = await axios.post(`${BASE_URL}/api/analyze/image`, form, {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 60000,
    });
    return res.data;
  } catch (err) {
    // Make network errors easier to debug
    const message =
      err.code === "ERR_NETWORK"
        ? `Network error. Is the API at ${BASE_URL} up?`
        : err.message;
    throw new Error(message, { cause: err });
  }
}

export async function analyzeAudio({ audio, text }) {
  const form = new FormData();
  form.append("text", text);
  form.append("audio", audio);
  try {
    const res = await axios.post(`${BASE_URL}/api/analyze/audio`, form, {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 60000,
    });
    return res.data;
  } catch (err) {
    const message =
      err.code === "ERR_NETWORK"
        ? `Network error. Is the API at ${BASE_URL} up?`
        : err.message;
    throw new Error(message, { cause: err });
  }
}

export async function analyzeVideo({ video, text, visual_weight }) {
  const form = new FormData();
  form.append("text", text);
  form.append("visual_weight", String(visual_weight ?? 0.55));
  form.append("video", video);
  try {
    const res = await axios.post(`${BASE_URL}/api/analyze/video`, form, {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 600000, // 10-minute timeout to match the backend
    });
    return res.data;
  } catch (err) {
    const message =
      err.code === "ERR_NETWORK"
        ? `Network error. Is the API at ${BASE_URL} up?`
        : err.message;
    throw new Error(message, { cause: err });
  }
}
