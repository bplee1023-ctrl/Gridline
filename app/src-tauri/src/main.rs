// Gridline: a thin native shell. All grading happens in the cloud pipeline; this app only
// shows the published snapshot, caches it on disk, and installs signed app updates.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{fs, path::PathBuf};

use tauri::Manager;
use tauri_plugin_updater::UpdaterExt;

struct UpdaterEnabled(bool);

/// ~/Library/Application Support/Gridline on macOS (the app's data dir elsewhere).
fn cache_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let dir = if cfg!(target_os = "macos") {
        app.path().home_dir().map_err(|e| e.to_string())?.join("Library/Application Support/Gridline")
    } else {
        app.path().app_data_dir().map_err(|e| e.to_string())?
    };
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// Snapshot paths like "2026/players/00-0037834.json" become flat, safe file names.
fn key_path(app: &tauri::AppHandle, key: &str) -> Result<PathBuf, String> {
    let safe: String = key
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' { c } else { '~' })
        .collect();
    if safe.is_empty() || safe.starts_with('.') {
        return Err("invalid cache key".into());
    }
    Ok(cache_dir(app)?.join(safe))
}

#[tauri::command]
fn cache_read(app: tauri::AppHandle, key: String) -> Result<Option<String>, String> {
    let p = key_path(&app, &key)?;
    match fs::read_to_string(&p) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

#[tauri::command]
fn cache_write(app: tauri::AppHandle, key: String, data: String) -> Result<(), String> {
    let p = key_path(&app, &key)?;
    let tmp = p.with_extension("tmp");
    fs::write(&tmp, data).map_err(|e| e.to_string())?;
    fs::rename(&tmp, &p).map_err(|e| e.to_string())
}

#[tauri::command]
async fn check_update(app: tauri::AppHandle) -> Result<Option<String>, String> {
    if !app.state::<UpdaterEnabled>().0 {
        return Ok(None);
    }
    let updater = app.updater().map_err(|e| e.to_string())?;
    match updater.check().await {
        Ok(Some(update)) => Ok(Some(update.version.clone())),
        Ok(None) => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

#[tauri::command]
async fn install_update(app: tauri::AppHandle) -> Result<(), String> {
    if !app.state::<UpdaterEnabled>().0 {
        return Err("updates are not configured for this build".into());
    }
    let updater = app.updater().map_err(|e| e.to_string())?;
    if let Some(update) = updater.check().await.map_err(|e| e.to_string())? {
        update
            .download_and_install(|_, _| {}, || {})
            .await
            .map_err(|e| e.to_string())?;
        app.restart();
    }
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            // The updater is only configured on release builds that carry a signing key.
            let enabled = app.config().plugins.0.contains_key("updater");
            if enabled {
                app.handle().plugin(tauri_plugin_updater::Builder::new().build())?;
            }
            app.manage(UpdaterEnabled(enabled));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![cache_read, cache_write, check_update, install_update])
        .run(tauri::generate_context!())
        .expect("error while running Gridline");
}
