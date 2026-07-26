"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import CropEditor, { Point } from "./CropEditor";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const PREVIEW_DEBOUNCE_MS = 300;

type Stage = "upload" | "edit";
type EnhanceMode = "auto" | "gray" | "bw" | "color";

interface DocState {
  quad: Point[];
  enhanceMode: EnhanceMode;
  strength: number;
  manualRotation: number;
  resultUrl: string | null;
}

function makeDoc(quad: Point[]): DocState {
  return { quad, enhanceMode: "auto", strength: 50, manualRotation: 0, resultUrl: null };
}

export default function Home() {
  const [stage, setStage] = useState<Stage>("upload");
  const [batchMode, setBatchMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewSize, setPreviewSize] = useState({ width: 0, height: 0 });
  const [usedFallback, setUsedFallback] = useState(false);

  const [documents, setDocuments] = useState<DocState[]>([]);
  const [activeDoc, setActiveDoc] = useState(0);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const resultUrlsRef = useRef<Map<number, string>>(new Map());

  const current = documents[activeDoc];

  function updateActiveDoc(patch: Partial<DocState>) {
    setDocuments((docs) => docs.map((d, i) => (i === activeDoc ? { ...d, ...patch } : d)));
  }

  async function handleFileSelected(file: File) {
    setError(null);
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const endpoint = batchMode ? "/api/detect_batch" : "/api/detect";
      const res = await fetch(`${API_URL}${endpoint}`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Detection failed (${res.status})`);
      const data = await res.json();

      const quads: Point[][] = batchMode ? data.quads : [data.quad];
      if (batchMode && quads.length === 0) {
        throw new Error("No documents detected in this photo — try single-document mode or adjust the shot.");
      }

      setSessionId(data.session_id);
      setUsedFallback(batchMode ? false : data.used_fallback);
      setPreviewSize({ width: data.resized_width, height: data.resized_height });
      setPreviewUrl(`${API_URL}/api/preview/${data.session_id}`);
      setDocuments(quads.map(makeDoc));
      setActiveDoc(0);
      resultUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
      resultUrlsRef.current.clear();
      setStage("edit");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  // Re-runs /api/enhance for one document (by index) with whatever crop/mode/
  // strength/rotation is current for it, called on mount and after debounced changes
  // so the preview always reflects the latest controls without a separate "generate"
  // button. Each document's result is cached by index so switching between documents
  // in batch mode doesn't lose or mix up previews.
  const runEnhance = useCallback(
    async (targetSessionId: string, docIndex: number, doc: DocState) => {
      if (doc.quad.length !== 4) return;
      setPreviewLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_URL}/api/enhance`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: targetSessionId,
            quad: doc.quad,
            enhance_mode: doc.enhanceMode,
            enhance_strength: doc.strength,
            manual_rotation: doc.manualRotation,
            doc_index: docIndex,
          }),
        });
        if (!res.ok) throw new Error(`Enhancement failed (${res.status})`);
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const prevUrl = resultUrlsRef.current.get(docIndex);
        if (prevUrl) URL.revokeObjectURL(prevUrl);
        resultUrlsRef.current.set(docIndex, url);
        setDocuments((docs) => docs.map((d, i) => (i === docIndex ? { ...d, resultUrl: url } : d)));
      } catch (e) {
        setError(e instanceof Error ? e.message : "Something went wrong");
      } finally {
        setPreviewLoading(false);
      }
    },
    []
  );

  // Debounced: crop dragging and the strength slider both fire rapidly, and calling
  // /api/enhance on every intermediate value would flood the backend and make the
  // preview lag behind the controls instead of reflecting where the user settled.
  useEffect(() => {
    if (stage !== "edit" || !sessionId || !current || current.quad.length !== 4) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    const docIndex = activeDoc;
    const doc = current;
    debounceRef.current = setTimeout(() => {
      runEnhance(sessionId, docIndex, doc);
    }, PREVIEW_DEBOUNCE_MS);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, current?.quad, current?.enhanceMode, current?.strength, current?.manualRotation, activeDoc, stage]);

  async function handleDownload(format: string, docIndex: number) {
    if (!sessionId) return;
    const res = await fetch(`${API_URL}/api/export/${sessionId}?format=${format}&doc_index=${docIndex}`);
    if (!res.ok) {
      setError(`Export failed (${res.status})`);
      return;
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = documents.length > 1 ? `scan_${docIndex + 1}.${format}` : `scan.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function reset() {
    setStage("upload");
    setSessionId(null);
    setPreviewUrl(null);
    resultUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    resultUrlsRef.current.clear();
    setDocuments([]);
    setActiveDoc(0);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 flex flex-col items-center px-4 py-10">
      <div className="w-full max-w-5xl flex flex-col gap-6">
        <header className="text-center">
          <h1 className="text-2xl font-semibold">CamScanner</h1>
          <p className="text-neutral-400 text-sm mt-1">
            Upload a photo of a document, adjust the crop, and export an enhanced scan.
          </p>
        </header>

        {error && (
          <div className="bg-red-950 border border-red-800 text-red-200 text-sm rounded-lg px-4 py-3">
            {error}
          </div>
        )}

        {stage === "upload" && (
          <div className="border-2 border-dashed border-neutral-700 rounded-xl p-12 flex flex-col items-center gap-4">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileSelected(file);
              }}
            />
            <label className="flex items-center gap-2 text-sm text-neutral-400 cursor-pointer">
              <input
                type="checkbox"
                checked={batchMode}
                onChange={(e) => setBatchMode(e.target.checked)}
                className="accent-cyan-600"
              />
              Batch scan: detect multiple documents in one photo
              <span className="text-amber-400">(experimental)</span>
            </label>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              className="bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-medium px-6 py-3 rounded-lg transition-colors"
            >
              {loading ? "Detecting document..." : "Choose photo"}
            </button>
          </div>
        )}

        {stage === "edit" && previewUrl && current && (
          <div className="flex flex-col gap-4">
            {usedFallback && (
              <p className="text-amber-400 text-sm">
                Couldn&apos;t confidently detect the document edges — adjust the corners below.
              </p>
            )}

            {documents.length > 1 && (
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-sm text-neutral-400">
                  {documents.length} documents detected (experimental — please review each crop):
                </span>
                {documents.map((_, i) => (
                  <button
                    key={i}
                    onClick={() => setActiveDoc(i)}
                    className={`px-3 py-1 rounded-md text-sm ${
                      activeDoc === i ? "bg-cyan-600 text-white" : "bg-neutral-800 text-neutral-300"
                    }`}
                  >
                    Doc {i + 1}
                  </button>
                ))}
              </div>
            )}

            <div className="grid md:grid-cols-2 gap-4">
              <div className="flex flex-col gap-2">
                <span className="text-sm text-neutral-400">Crop</span>
                <CropEditor
                  imageUrl={previewUrl}
                  imageWidth={previewSize.width}
                  imageHeight={previewSize.height}
                  quad={current.quad}
                  onChange={(q) => updateActiveDoc({ quad: q })}
                />
              </div>

              <div className="flex flex-col gap-2">
                <span className="text-sm text-neutral-400">Preview {previewLoading && "(updating...)"}</span>
                <div className="rounded-lg border border-neutral-700 bg-neutral-900 min-h-[200px] flex items-center justify-center overflow-hidden">
                  {current.resultUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={current.resultUrl} alt="Enhanced scan preview" className="max-w-full max-h-[520px] object-contain" />
                  ) : (
                    <span className="text-neutral-600 text-sm">Generating preview...</span>
                  )}
                </div>
              </div>
            </div>

            <div className="flex flex-col gap-3 bg-neutral-900 rounded-lg p-4">
              <div className="flex items-center gap-3">
                <span className="text-sm text-neutral-400 w-24">Look</span>
                <div className="flex gap-2">
                  {(["auto", "gray", "bw", "color"] as EnhanceMode[]).map((m) => (
                    <button
                      key={m}
                      onClick={() => updateActiveDoc({ enhanceMode: m })}
                      className={`px-3 py-1.5 rounded-md text-sm ${
                        current.enhanceMode === m ? "bg-cyan-600 text-white" : "bg-neutral-800 text-neutral-300"
                      }`}
                    >
                      {m === "auto" ? "Auto" : m === "gray" ? "Gray" : m === "bw" ? "Black & white" : "Color"}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex items-center gap-3">
                <span className="text-sm text-neutral-400 w-24">Strength</span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={current.strength}
                  onChange={(e) => updateActiveDoc({ strength: Number(e.target.value) })}
                  className="flex-1"
                />
                <span className="text-sm text-neutral-400 w-8 text-right">{current.strength}</span>
              </div>

              <div className="flex items-center gap-3">
                <span className="text-sm text-neutral-400 w-24">Rotate</span>
                <button
                  onClick={() => updateActiveDoc({ manualRotation: (current.manualRotation + 270) % 360 })}
                  className="bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-sm px-3 py-1.5 rounded-md"
                  title="Rotate 90° counter-clockwise"
                >
                  ⟲ Left
                </button>
                <button
                  onClick={() => updateActiveDoc({ manualRotation: (current.manualRotation + 90) % 360 })}
                  className="bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-sm px-3 py-1.5 rounded-md"
                  title="Rotate 90° clockwise"
                >
                  ⟳ Right
                </button>
                {current.manualRotation !== 0 && (
                  <span className="text-sm text-neutral-500">{current.manualRotation}°</span>
                )}
              </div>
            </div>

            <div className="flex flex-col gap-3">
              <div className="flex flex-wrap gap-2">
                {["png", "jpg", "pdf", "tiff", "webp"].map((fmt) => (
                  <button
                    key={fmt}
                    onClick={() => handleDownload(fmt, activeDoc)}
                    disabled={!current.resultUrl}
                    className="bg-neutral-800 hover:bg-neutral-700 disabled:opacity-40 text-neutral-200 text-sm px-4 py-2 rounded-md"
                  >
                    Download .{fmt}
                  </button>
                ))}
              </div>
              <button
                onClick={reset}
                className="bg-neutral-800 hover:bg-neutral-700 text-neutral-200 font-medium px-6 py-3 rounded-lg transition-colors"
              >
                Scan another
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
