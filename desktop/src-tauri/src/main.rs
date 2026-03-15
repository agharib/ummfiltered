#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    env, fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use rfd::FileDialog;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{path::BaseDirectory, AppHandle, Emitter, Manager, State};

#[derive(Default)]
struct WorkerState {
    child: Mutex<Option<Child>>,
    request_file: Mutex<Option<PathBuf>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DesktopOverrides {
    model_size: Option<String>,
    quality: Option<String>,
    min_confidence: Option<f64>,
    custom_fillers: Option<Vec<String>>,
    fixed_pause_ms: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DesktopProcessRequest {
    input_path: String,
    output_path: Option<String>,
    preset: String,
    aggressive: bool,
    verify_pass: bool,
    natural_pauses: bool,
    overrides: DesktopOverrides,
}

struct WorkerLaunch {
    command: PathBuf,
    args: Vec<String>,
    current_dir: PathBuf,
    extra_env: Vec<(String, String)>,
}

#[tauri::command]
fn pick_input_file() -> Result<Option<String>, String> {
    Ok(FileDialog::new()
        .add_filter("Video", &["mp4", "mov", "m4v", "avi", "mkv"])
        .pick_file()
        .map(|path| path.display().to_string()))
}

#[tauri::command]
fn pick_output_file(
    default_name: String,
    default_dir: Option<String>,
) -> Result<Option<String>, String> {
    let dialog = if let Some(dir) = default_dir {
        FileDialog::new().set_directory(dir)
    } else {
        FileDialog::new()
    };

    Ok(dialog
        .set_file_name(default_name)
        .save_file()
        .map(|path| path.display().to_string()))
}

#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    run_macos_open(["open", &path])
}

#[tauri::command]
fn reveal_in_finder(path: String) -> Result<(), String> {
    run_macos_open(["open", "-R", &path])
}

#[tauri::command]
fn cancel_pipeline_job(state: State<'_, Arc<WorkerState>>) -> Result<(), String> {
    let mut child_guard = state
        .child
        .lock()
        .map_err(|_| "Worker state is unavailable.".to_string())?;
    if let Some(child) = child_guard.as_mut() {
        let pid = child.id().to_string();
        let status = Command::new("kill")
            .arg("-TERM")
            .arg(&pid)
            .status()
            .map_err(|error| format!("Failed to request worker cancellation: {error}"))?;
        if !status.success() {
            child
                .kill()
                .map_err(|error| format!("Failed to force-cancel worker: {error}"))?;
        }
    }
    Ok(())
}

#[tauri::command]
fn start_pipeline_job(
    app: AppHandle,
    state: State<'_, Arc<WorkerState>>,
    request: DesktopProcessRequest,
) -> Result<(), String> {
    {
        let child_guard = state
            .child
            .lock()
            .map_err(|_| "Worker state is unavailable.".to_string())?;
        if child_guard.is_some() {
            return Err("A processing job is already running.".to_string());
        }
    }

    let request_path = write_request_file(&request)?;
    {
        let mut request_guard = state
            .request_file
            .lock()
            .map_err(|_| "Worker request state is unavailable.".to_string())?;
        *request_guard = Some(request_path.clone());
    }

    let launch = resolve_worker_launch(&app, &request_path)?;

    let mut child = Command::new(&launch.command);
    child
        .current_dir(&launch.current_dir)
        .args(&launch.args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    for (key, value) in &launch.extra_env {
        child.env(key, value);
    }

    let mut child = child
        .spawn()
        .map_err(|error| format!("Failed to start Python worker: {error}"))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Failed to capture worker stdout.".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Failed to capture worker stderr.".to_string())?;

    {
        let mut child_guard = state
            .child
            .lock()
            .map_err(|_| "Worker state is unavailable.".to_string())?;
        *child_guard = Some(child);
    }

    spawn_stdout_thread(app.clone(), stdout);
    spawn_stderr_thread(app.clone(), stderr);
    spawn_exit_thread(app, state.inner().clone());

    Ok(())
}

fn resolve_worker_launch(app: &AppHandle, request_path: &Path) -> Result<WorkerLaunch, String> {
    let app_support_dir = ensure_app_support_dir(app)?;

    if let Some(worker_path) = resolve_packaged_worker_path(app)? {
        let worker_dir = worker_path
            .parent()
            .ok_or_else(|| "Bundled worker directory is invalid.".to_string())?
            .to_path_buf();
        return Ok(WorkerLaunch {
            command: worker_path,
            args: vec![
                "--request-file".to_string(),
                request_path.display().to_string(),
            ],
            current_dir: worker_dir,
            extra_env: vec![(
                "UMMFILTERED_APP_SUPPORT".to_string(),
                app_support_dir.display().to_string(),
            )],
        });
    }

    let workspace_root = workspace_root()?;
    let python = env::var("UMMFILTERED_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let python_path = env::var("PYTHONPATH")
        .map(|existing| format!("{existing}:{}", workspace_root.display()))
        .unwrap_or_else(|_| workspace_root.display().to_string());

    Ok(WorkerLaunch {
        command: PathBuf::from(python),
        args: vec![
            "-m".to_string(),
            "ummfiltered.gui_worker".to_string(),
            "--request-file".to_string(),
            request_path.display().to_string(),
        ],
        current_dir: workspace_root,
        extra_env: vec![
            ("PYTHONPATH".to_string(), python_path),
            (
                "UMMFILTERED_APP_SUPPORT".to_string(),
                app_support_dir.display().to_string(),
            ),
        ],
    })
}

fn resolve_packaged_worker_path(app: &AppHandle) -> Result<Option<PathBuf>, String> {
    if let Ok(override_path) = env::var("UMMFILTERED_WORKER_PATH") {
        let path = PathBuf::from(override_path);
        if path.exists() {
            return Ok(Some(path));
        }
        return Err(
            "UMMFILTERED_WORKER_PATH was set but the worker binary was not found.".to_string(),
        );
    }

    let resource_path = app
        .path()
        .resolve(
            "worker/ummfiltered-gui-worker/ummfiltered-gui-worker",
            BaseDirectory::Resource,
        )
        .map_err(|error| format!("Failed to resolve bundled worker path: {error}"))?;

    if resource_path.exists() {
        return Ok(Some(resource_path));
    }

    Ok(None)
}

fn ensure_app_support_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let app_support_dir = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Failed to resolve app data directory: {error}"))?;
    fs::create_dir_all(&app_support_dir)
        .map_err(|error| format!("Failed to create app data directory: {error}"))?;
    Ok(app_support_dir)
}

fn run_macos_open<const N: usize>(args: [&str; N]) -> Result<(), String> {
    Command::new(args[0])
        .args(&args[1..])
        .status()
        .map_err(|error| format!("Failed to launch macOS open command: {error}"))
        .and_then(|status| {
            if status.success() {
                Ok(())
            } else {
                Err("macOS open command exited unsuccessfully.".to_string())
            }
        })
}

fn workspace_root() -> Result<PathBuf, String> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .map_err(|error| format!("Failed to resolve workspace root: {error}"))
}

fn write_request_file(request: &DesktopProcessRequest) -> Result<PathBuf, String> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("Failed to build request timestamp: {error}"))?
        .as_millis();
    let path = env::temp_dir().join(format!("ummfiltered-desktop-{timestamp}.json"));
    let payload = serde_json::to_string(request)
        .map_err(|error| format!("Failed to serialize request: {error}"))?;
    fs::write(&path, payload).map_err(|error| format!("Failed to write request file: {error}"))?;
    Ok(path)
}

fn spawn_stdout_thread(app: AppHandle, stdout: impl std::io::Read + Send + 'static) {
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if line.trim().is_empty() {
                continue;
            }
            let payload: Value = serde_json::from_str(&line).unwrap_or_else(|_| {
                json!({
                    "type": "error",
                    "code": "invalid_json",
                    "message": "Worker emitted invalid JSON.",
                    "details": line,
                })
            });
            let _ = app.emit("pipeline://message", payload);
        }
    });
}

fn spawn_stderr_thread(app: AppHandle, stderr: impl std::io::Read + Send + 'static) {
    thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if line.trim().is_empty() {
                continue;
            }
            let _ = app.emit(
                "pipeline://message",
                json!({
                    "type": "event",
                    "kind": "warning",
                    "stage": null,
                    "message": line,
                }),
            );
        }
    });
}

fn spawn_exit_thread(app: AppHandle, state: Arc<WorkerState>) {
    thread::spawn(move || loop {
        let mut emit_exit = None;
        let mut emit_error = None;

        {
            let mut child_guard = match state.child.lock() {
                Ok(guard) => guard,
                Err(_) => return,
            };

            let Some(child) = child_guard.as_mut() else {
                return;
            };

            match child.try_wait() {
                Ok(Some(status)) => {
                    *child_guard = None;
                    cleanup_request_file(&state);
                    emit_exit = Some(status.code());
                }
                Ok(None) => {}
                Err(error) => {
                    *child_guard = None;
                    cleanup_request_file(&state);
                    emit_error = Some(error.to_string());
                    emit_exit = Some(None);
                }
            }
        }

        if let Some(error) = emit_error {
            let _ = app.emit(
                "pipeline://message",
                json!({
                    "type": "error",
                    "code": "worker_wait_failed",
                    "message": "The desktop worker could not be monitored cleanly.",
                    "details": error,
                }),
            );
        }

        if let Some(code) = emit_exit {
            let _ = app.emit("pipeline://exit", json!({ "code": code }));
            return;
        }

        thread::sleep(Duration::from_millis(120));
    });
}

fn cleanup_request_file(state: &Arc<WorkerState>) {
    let path = {
        let mut request_guard = match state.request_file.lock() {
            Ok(guard) => guard,
            Err(_) => return,
        };
        request_guard.take()
    };

    if let Some(path) = path {
        let _ = fs::remove_file(path);
    }
}

fn main() {
    tauri::Builder::default()
        .manage(Arc::new(WorkerState::default()))
        .invoke_handler(tauri::generate_handler![
            pick_input_file,
            pick_output_file,
            open_path,
            reveal_in_finder,
            start_pipeline_job,
            cancel_pipeline_job
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
