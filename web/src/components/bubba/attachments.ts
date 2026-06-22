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
