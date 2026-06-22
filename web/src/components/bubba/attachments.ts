// Browser-only attachment helpers for Bubba (kept out of Bubba.tsx so the
// component stays focused, and so pdfjs-dist is dynamically imported on demand).

/** Read an image File as a data: URL (for an image attachment). */
export function readImageFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error ?? new Error("file read failed"));
    reader.readAsDataURL(file);
  });
}

/** Capture one frame of a user-chosen screen/window as a PNG data URL. */
export async function captureScreen(): Promise<string> {
  const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
  try {
    const video = document.createElement("video");
    video.srcObject = stream;
    await new Promise<void>((resolve) => {
      video.onloadedmetadata = () => resolve();
    });
    await video.play();
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("no 2d canvas context");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    video.pause();
    return canvas.toDataURL("image/png");
  } finally {
    stream.getTracks().forEach((t) => t.stop()); // always release the capture
  }
}

const PDF_TEXT_CAP = 12000;

/** Extract text from a PDF File in the browser (capped). Returns "" on no text. */
export async function extractPdfText(file: File): Promise<string> {
  const pdfjs = await import("pdfjs-dist");
  // Worker: the bundler resolves the worker asset from the package via new URL().
  pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).href;
  const data = new Uint8Array(await file.arrayBuffer());
  const doc = await pdfjs.getDocument({ data }).promise;
  let text = "";
  for (let p = 1; p <= doc.numPages; p++) {
    const page = await doc.getPage(p);
    const content = await page.getTextContent();
    text += content.items.map((it) => ("str" in it ? it.str : "")).join(" ") + "\n";
    if (text.length >= PDF_TEXT_CAP) break;
  }
  text = text.trim();
  return text.length > PDF_TEXT_CAP ? text.slice(0, PDF_TEXT_CAP) + "\n…(truncated)" : text;
}
