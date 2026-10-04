// Export every docs/assets/excalidraw/*.excalidraw to a sibling .svg with fonts embedded.
//
//   cd tools/excalidraw_export && npm install && node export.mjs          (all drawings)
//   node export.mjs gqa mla                                              (only these)
//
// Uses the same Excalidraw build as docs/assets/viewer/excalidraw.html, run in a locally
// installed Chrome/Edge through playwright-core (set CHROME_PATH to override the browser).
import { chromium } from "playwright-core";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const EXCALIDRAW = "https://unpkg.com/@excalidraw/excalidraw@0.17.6";
const DIR = resolve(dirname(fileURLToPath(import.meta.url)), "../../docs/assets/excalidraw");

const BROWSERS = [
  process.env.CHROME_PATH,
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "/usr/bin/google-chrome",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].filter(Boolean);
const executablePath = BROWSERS.find((p) => existsSync(p));
if (!executablePath) throw new Error("No Chrome/Edge found; set CHROME_PATH.");

const wanted = process.argv.slice(2);
const files = readdirSync(DIR)
  .filter((f) => f.endsWith(".excalidraw"))
  .filter((f) => !wanted.length || wanted.includes(basename(f, ".excalidraw")));

const fontCache = new Map();
async function dataUri(url) {
  if (!fontCache.has(url)) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`font ${url}: HTTP ${res.status}`);
    const b64 = Buffer.from(await res.arrayBuffer()).toString("base64");
    fontCache.set(url, `data:font/woff2;base64,${b64}`);
  }
  return fontCache.get(url);
}

const browser = await chromium.launch({ executablePath });
const page = await browser.newPage();
await page.setContent(`<!doctype html><html><head>
  <script>window.EXCALIDRAW_ASSET_PATH = "${EXCALIDRAW}";</script>
  <script src="https://unpkg.com/react@18.2.0/umd/react.production.min.js"></script>
  <script src="https://unpkg.com/react-dom@18.2.0/umd/react-dom.production.min.js"></script>
  <script src="${EXCALIDRAW}/dist/excalidraw.production.min.js"></script>
  </head><body></body></html>`, { waitUntil: "load" });
await page.waitForFunction(() => !!window.ExcalidrawLib);

for (const f of files) {
  const scene = JSON.parse(readFileSync(join(DIR, f), "utf8"));
  let svg = await page.evaluate(async (scene) => {
    const el = await window.ExcalidrawLib.exportToSvg({
      elements: scene.elements.filter((e) => !e.isDeleted),
      appState: {
        exportBackground: true,
        viewBackgroundColor: scene.appState?.viewBackgroundColor || "#ffffff",
        exportWithDarkMode: false,
        exportPadding: 24,
      },
      files: scene.files || {},
    });
    return el.outerHTML;
  }, scene);
  for (const url of new Set(svg.match(/https?:[^"')\s]+\.woff2/g) || [])) {
    svg = svg.split(url).join(await dataUri(url));
  }
  const out = join(DIR, f.replace(/\.excalidraw$/, ".svg"));
  writeFileSync(out, svg);
  console.log(`${basename(out)}  ${(svg.length / 1024).toFixed(0)} KiB`);
}
await browser.close();
