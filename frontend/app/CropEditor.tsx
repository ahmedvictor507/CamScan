"use client";

import { useEffect, useRef, useState } from "react";

export type Point = [number, number];

interface CropEditorProps {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  quad: Point[];
  onChange: (quad: Point[]) => void;
}

const HANDLE_RADIUS = 10;

// Renders the resized-space photo with the 4-point quad overlaid as draggable
// handles. Coordinates are kept in the backend's resized-image space throughout
// (the space /api/detect returned the quad in) and only converted to/from canvas
// pixels for drawing and hit-testing, so the quad sent back to /api/enhance needs
// no further transformation.
export default function CropEditor({ imageUrl, imageWidth, imageHeight, quad, onChange }: CropEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [displayScale, setDisplayScale] = useState(1);

  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      draw();
    };
    img.src = imageUrl;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageUrl]);

  useEffect(() => {
    draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [quad, displayScale]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const containerWidth = canvas.parentElement?.clientWidth ?? imageWidth;
    const scale = Math.min(1, containerWidth / imageWidth);
    setDisplayScale(scale);
  }, [imageWidth]);

  function draw() {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = imageWidth * displayScale;
    canvas.height = imageHeight * displayScale;
    // Pin the CSS display size to the canvas's own pixel buffer -- without this,
    // `max-w-full` lets the browser stretch the element to fill its flex container,
    // which desyncs pointer coordinates (measured via getBoundingClientRect, in CSS
    // pixels) from the buffer coordinates the quad/hit-testing math is written in.
    canvas.style.width = `${canvas.width}px`;
    canvas.style.height = `${canvas.height}px`;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const pts = quad.map(([x, y]) => [x * displayScale, y * displayScale]);

    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
    ctx.closePath();
    ctx.stroke();

    pts.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x, y, HANDLE_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = "#22d3ee";
      ctx.fill();
      ctx.strokeStyle = "#0e7490";
      ctx.lineWidth = 2;
      ctx.stroke();
    });
  }

  function toCanvasCoords(e: React.PointerEvent<HTMLCanvasElement>): Point {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  }

  function handlePointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    const [cx, cy] = toCanvasCoords(e);
    const idx = quad.findIndex(([x, y]) => {
      const dx = x * displayScale - cx;
      const dy = y * displayScale - cy;
      return Math.sqrt(dx * dx + dy * dy) <= HANDLE_RADIUS + 6;
    });
    if (idx !== -1) {
      setDragIndex(idx);
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    }
  }

  function handlePointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    if (dragIndex === null) return;
    const [cx, cy] = toCanvasCoords(e);
    const x = Math.max(0, Math.min(imageWidth, cx / displayScale));
    const y = Math.max(0, Math.min(imageHeight, cy / displayScale));
    const next = quad.slice() as Point[];
    next[dragIndex] = [x, y];
    onChange(next);
  }

  function handlePointerUp() {
    setDragIndex(null);
  }

  return (
    <canvas
      ref={canvasRef}
      className="max-w-full touch-none rounded-lg border border-neutral-700 cursor-crosshair"
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    />
  );
}
