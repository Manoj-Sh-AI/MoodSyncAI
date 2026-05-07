import express from "express";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";

const router = express.Router();

// Per-modality uploaders with sensible defaults; override via env if needed
const uploadImage = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: (Number(process.env.IMAGE_MAX_UPLOAD_MB) || 12) * 1024 * 1024,
  },
});

const uploadAudio = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: (Number(process.env.AUDIO_MAX_UPLOAD_MB) || 30) * 1024 * 1024,
  },
});

const uploadVideo = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: (Number(process.env.VIDEO_MAX_UPLOAD_MB) || 200) * 1024 * 1024,
  },
});

const PY_SERVICE_URL = process.env.PY_SERVICE_URL || "http://localhost:8001";

// POST /api/analyze/image
router.post("/image", uploadImage.single("image"), async (req, res) => {
  try {
    const { text, visual_weight } = req.body;
    if (!text || !text.trim()) {
      return res.status(400).json({ error: "Missing text" });
    }
    if (!req.file) {
      return res.status(400).json({ error: "Missing image file" });
    }

    const form = new FormData();
    form.append("text", text);
    form.append("visual_weight", String(visual_weight || 0.55));
    form.append("image", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    const response = await axios.post(`${PY_SERVICE_URL}/analyze`, form, {
      headers: form.getHeaders(),
      timeout: Number(process.env.PY_SERVICE_TIMEOUT_MS || 180000),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });

    res.json(response.data);
  } catch (err) {
    console.error("/api/analyze/image proxy error", {
      code: err.code,
      message: err.message,
    });
    const status = err.response?.status || 500;
    res
      .status(status)
      .json(err.response?.data || { error: "Python service error" });
  }
});

// POST /api/analyze/audio
router.post("/audio", uploadAudio.single("audio"), async (req, res) => {
  try {
    const { text } = req.body;
    if (!text || !text.trim()) {
      return res.status(400).json({ error: "Missing text" });
    }
    if (!req.file) {
      return res.status(400).json({ error: "Missing audio file" });
    }

    const form = new FormData();
    form.append("text", text);
    form.append("audio", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    const response = await axios.post(`${PY_SERVICE_URL}/analyze_audio`, form, {
      headers: form.getHeaders(),
      timeout: Number(process.env.PY_SERVICE_TIMEOUT_MS || 180000),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });

    res.json(response.data);
  } catch (err) {
    console.error("/api/analyze/audio proxy error", {
      code: err.code,
      message: err.message,
    });
    const status = err.response?.status || 500;
    res
      .status(status)
      .json(err.response?.data || { error: "Python service error" });
  }
});

// POST /api/analyze/video
router.post("/video", uploadVideo.single("video"), async (req, res) => {
  try {
    const { text, visual_weight } = req.body;
    if (!text || !text.trim()) {
      return res.status(400).json({ error: "Missing text" });
    }
    if (!req.file) {
      return res.status(400).json({ error: "Missing video file" });
    }

    const form = new FormData();
    form.append("text", text);
    form.append("visual_weight", String(visual_weight || 0.55));
    form.append("video", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    const response = await axios.post(`${PY_SERVICE_URL}/analyze_video`, form, {
      headers: form.getHeaders(),
      timeout: Number(process.env.PY_SERVICE_TIMEOUT_MS || 180000),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });

    res.json(response.data);
  } catch (err) {
    console.error("/api/analyze/video proxy error", {
      code: err.code,
      message: err.message,
    });
    const status = err.response?.status || 500;
    res
      .status(status)
      .json(err.response?.data || { error: "Python service error" });
  }
});

// Handle Multer errors (e.g., file too large)
router.use((err, _req, res, next) => {
  if (err instanceof multer.MulterError) {
    return res.status(413).json({ error: `Upload error: ${err.message}` });
  }
  next(err);
});

export default router;
