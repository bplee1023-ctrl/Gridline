# Gridline setup guide

About 45 minutes the first time, most of it waiting on GitHub. You need a GitHub account and
your Mac. Nothing here costs money.

## 1. Put the code on GitHub

1. Unzip `gridline.zip`. You get a folder called `gridline` that is already a git repository.
2. On github.com, create a new repository named **gridline**. Make it **Public**. Leave
   "Add a README" unchecked.
3. In Terminal:

   ```bash
   cd path/to/gridline
   git remote add origin https://github.com/YOUR-USERNAME/gridline.git
   git push -u origin main
   ```

   Want it private instead? Claim GitHub Pro free through the GitHub Student Developer Pack
   first; Pages on a private repo needs it.

## 2. Turn on Pages and Actions

1. Repo → **Settings → Pages** → under "Build and deployment", set **Source** to
   **GitHub Actions**.
2. Repo → **Settings → Actions → General** → "Workflow permissions" → choose
   **Read and write permissions** → **Save**.
3. Open the **Actions** tab. If GitHub asks, click **I understand my workflows, go ahead and
   enable them**.

## 3. Run the first grading

1. **Actions → Grade → Run workflow** (leave "Re-grade" checked) → **Run workflow**.
2. The first run takes 10–20 minutes. It builds the 2015–2025 modeling baseline, saves it as a
   release named `baseline`, grades 2026, runs the UI tests, then publishes.
3. When both jobs are green, open `https://YOUR-USERNAME.github.io/gridline/`. That's the
   web version of the app. If it loads rankings, the data side is done.

From now on it runs itself at :30 past 00, 06, 12 and 18 UTC, skips runs when nflverse hasn't
changed anything, and does a full rebuild every Thursday for stat corrections. If a run fails,
GitHub emails you and the app keeps showing the last good snapshot.

## 4. Install the Mac app

Pick one.

### Option A — build it yourself (no security prompt)

1. Install Apple's command line tools: `xcode-select --install`
2. Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` (accept the
   defaults, then open a new Terminal window).
3. Install Node.js 20 or newer from nodejs.org (or `brew install node`).
4. Build:

   ```bash
   cd path/to/gridline/app
   npm install
   npm run tauri build
   ```

   The first build takes 5–10 minutes. It reads your GitHub remote to find your data
   address automatically.
5. Open `app/src-tauri/target/release/bundle/dmg/`, double-click the `.dmg`, and drag
   **Gridline** to **Applications**.

### Option B — download from GitHub

1. In Terminal, from the `gridline` folder: `git tag v1.0.0 && git push origin v1.0.0`
2. Wait for **Actions → Release app** to finish (about 15 minutes).
3. Open the repo's **Releases** page, download `Gridline_1.0.0_universal.dmg`, open it and drag
   Gridline to Applications.
4. The first launch is blocked because the app isn't notarized. Open **System Settings →
   Privacy & Security**, scroll down, and click **Open Anyway** next to the Gridline message.
   You only do this once.

## 5. Optional: let the app update itself

Data updates are already automatic. This step makes new *app versions* install themselves too.
Skip it if you're happy rebuilding with Option A when you change the code.

1. Generate a free signing key (set a password when asked):

   ```bash
   cd path/to/gridline/app
   npx tauri signer generate -w ~/.tauri/gridline.key
   ```

2. Repo → **Settings → Secrets and variables → Actions**:
   * **Secrets** tab → New repository secret **TAURI_SIGNING_PRIVATE_KEY** = the full contents
     of `~/.tauri/gridline.key`.
   * New repository secret **TAURI_SIGNING_PRIVATE_KEY_PASSWORD** = the password you chose.
   * **Variables** tab → New repository variable **UPDATER_PUBKEY** = the full contents of
     `~/.tauri/gridline.key.pub`.
3. Release a version: `git tag v1.0.1 && git push origin v1.0.1`. Install that `.dmg` once
   (Option B). Every later tagged release downloads in the background and shows a "Restart to
   update" bar in the app.

Keep `~/.tauri/gridline.key` safe. If you lose it, installed copies can't verify new updates
and you'll reinstall once by hand.

## Day to day

* The toolbar shows what you're looking at: "Through Week 5 · FTN charting complete ·
  updated Tue 9:12 AM". Grades from games still waiting on FTN charting are tagged
  provisional and settle within about 48 hours.
* A player ranked at the wrong position (EDGE vs LB, slot CB vs S): add his ID to
  `config/position_overrides.yaml`, commit and push. His ID is in the player page's address.
* Methodology weights, priors and minimums live in `config/`. Editing them takes effect on the
  next run; use **Run workflow** to apply immediately.
* After the Super Bowl, run **Grade** by hand with **Rebuild the baseline** checked so 2026
  joins the history for 2027.

## If something looks wrong

| You see | Do this |
| --- | --- |
| App says "No data yet" | The first Grade run hasn't finished, or Pages isn't set to GitHub Actions (step 2.1). |
| App shows an old week | Check the Actions tab for a red run; GitHub also emails failures. |
| Grade run fails at "Grade" with a schema error | nflverse renamed a column. The message names the feed and column; update `pipeline/ingest.py`. |
| App points at the wrong address | Methodology page → **Data source** → paste `https://YOUR-USERNAME.github.io/gridline/` → **Save and reload**. |
| Scheduled runs stopped | GitHub pauses schedules after 60 quiet days. The Thursday run commits a keep-alive to prevent this; if it happened anyway, Actions → Grade → **Enable workflow**. |
