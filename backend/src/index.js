import express from "express";
import cors from "cors";
import morgan from "morgan";
import helmet from "helmet";
import dotenv from "dotenv";
import rateLimit from "express-rate-limit";
import analyzeRouter from "./routes/analyze.js";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 8080;

app.use(helmet());
// Robust CORS: allow '*' or a comma-separated whitelist from env
const corsOrigin = (() => {
  const raw = process.env.CORS_ORIGIN;
  if (!raw || raw.trim() === "" || raw.trim() === "*") return "*";
  return raw.split(",").map((s) => s.trim());
})();
app.use(cors({ origin: corsOrigin, credentials: false }));
app.options("*", cors({ origin: corsOrigin }));
app.use(express.json({ limit: "5mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(morgan("dev"));

const limiter = rateLimit({ windowMs: 60 * 1000, max: 100 });
app.use("/api/", limiter);

app.get("/", (_req, res) => res.json({ status: "ok" }));
app.use("/api/analyze", analyzeRouter);

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: "Not Found", path: req.originalUrl });
});

// Error handler
app.use((err, _req, res, _next) => {
  console.error(err);
  res.status(500).json({ error: "Internal Server Error" });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`API server listening on http://0.0.0.0:${PORT}`);
});
