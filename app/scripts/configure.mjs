// Prepares the UI for a build:
//  1. Writes ui-dist/index.html with the GitHub Pages data address baked in.
//     Address = $GRIDLINE_DATA_BASE, else derived from this repo's `origin` remote
//     (github.com/<owner>/<repo>  ->  https://<owner>.github.io/<repo>/).
//  2. When $GRIDLINE_UPDATER_PUBKEY is set (release builds), writes
//     src-tauri/tauri.updater.conf.json so the app checks GitHub Releases for signed updates.
import { execSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync, existsSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const app = join(here, "..");

function repoSlug() {
  if (process.env.GITHUB_REPOSITORY) return process.env.GITHUB_REPOSITORY;
  try {
    const url = execSync("git remote get-url origin", { cwd: app, stdio: ["ignore", "pipe", "ignore"] }).toString().trim();
    const m = url.match(/github\.com[:/]([^/]+)\/([^/]+?)(\.git)?$/);
    if (m) return `${m[1]}/${m[2]}`;
  } catch {}
  return null;
}

const slug = repoSlug();
let base = process.env.GRIDLINE_DATA_BASE || "";
if (!base && slug) {
  const [owner, repo] = slug.split("/");
  base = repo.toLowerCase() === `${owner.toLowerCase()}.github.io`
    ? `https://${owner.toLowerCase()}.github.io/`
    : `https://${owner.toLowerCase()}.github.io/${repo}/`;
}
if (!base) {
  console.warn("[configure] No data address found (no git remote and no GRIDLINE_DATA_BASE). " +
               "The app will ask for it on the Methodology page.");
}

const html = readFileSync(join(app, "ui", "index.html"), "utf8").replace("__GRIDLINE_DATA_BASE__", base);
mkdirSync(join(app, "ui-dist"), { recursive: true });
writeFileSync(join(app, "ui-dist", "index.html"), html);
console.log(`[configure] data address: ${base || "(unset)"}`);

const updaterConf = join(app, "src-tauri", "tauri.updater.conf.json");
const pubkey = process.env.GRIDLINE_UPDATER_PUBKEY;
if (pubkey && slug) {
  const conf = {
    bundle: { createUpdaterArtifacts: true },
    plugins: { updater: { pubkey, endpoints: [`https://github.com/${slug}/releases/latest/download/latest.json`] } },
  };
  if (process.env.GRIDLINE_VERSION) conf.version = process.env.GRIDLINE_VERSION.replace(/^v/, "");
  writeFileSync(updaterConf, JSON.stringify(conf, null, 2));
  console.log("[configure] auto-updater enabled");
} else if (existsSync(updaterConf) && !pubkey) {
  rmSync(updaterConf);
}
