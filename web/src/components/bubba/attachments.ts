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
