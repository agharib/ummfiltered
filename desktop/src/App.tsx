import { type ReactNode, useEffect, useId, useState } from "react";

import {
  PRESETS,
  STAGE_ORDER,
  applyPreset,
  buildGuiRequest,
  deriveOutputName,
  loadPersistedSettings,
  persistSettings,
  resolvedModelSize,
  resolvedQuality,
  splitPath,
} from "./lib/settings";
import {
  cancelPipelineJob,
  openPath,
  pickOutputFile,
  pickVideoFile,
  revealInFinder,
  startPipelineJob,
  subscribeToFileDrops,
  subscribeToPipelineMessages,
} from "./lib/tauri";
import type {
  PauseMode,
  PipelineResult,
  PipelineStage,
  PresetKey,
  UiSettings,
  WorkerLine,
} from "./lib/types";

type ScreenState = "setup" | "processing" | "success" | "error";

const STAGE_LABELS: Record<PipelineStage, string> = {
  probe: "Inspecting source",
  extract_audio: "Extracting audio",
  transcribe: "Transcribing speech",
  detect_fillers: "Finding fillers",
  plan_cuts: "Planning edits",
  render: "Rendering output",
  verify: "Verifying output",
  final_check: "Final quality check",
};

export default function App() {
  const [settings, setSettings] = useState<UiSettings>(() => loadPersistedSettings());
  const [screen, setScreen] = useState<ScreenState>("setup");
  const [activity, setActivity] = useState<string[]>([]);
  const [activeStage, setActiveStage] = useState<PipelineStage | null>(null);
  const [completedStages, setCompletedStages] = useState<PipelineStage[]>([]);
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [isDragActive, setIsDragActive] = useState(false);

  useEffect(() => {
    persistSettings(settings);
  }, [settings]);

  useEffect(() => {
    let offMessages: (() => void) | undefined;
    let offDrops: (() => void) | undefined;

    void subscribeToPipelineMessages(handleWorkerLine).then((dispose) => {
      offMessages = dispose;
    });
    void subscribeToFileDrops((path) => {
      setIsDragActive(false);
      applyInputPath(path);
    }).then((dispose) => {
      offDrops = dispose;
    });

    return () => {
      offMessages?.();
      offDrops?.();
    };
  }, []);

  const selectedPreset = PRESETS[settings.preset];
  const resolvedModel = resolvedModelSize(settings);
  const resolvedExport = resolvedQuality(settings);
  const canStart = Boolean(settings.inputPath) && screen !== "processing";
  const stageIndex = activeStage ? STAGE_ORDER.indexOf(activeStage) : -1;
  const processingProgress = Math.max(
    activity.length > 0 ? 0.16 : 0.1,
    stageIndex >= 0 ? (stageIndex + 1) / STAGE_ORDER.length : completedStages.length / STAGE_ORDER.length,
  );

  function handleWorkerLine(line: WorkerLine) {
    if (line.type === "event") {
      setActivity((previous) => [formatActivity(line), ...previous].slice(0, 18));
      if (line.stage && line.kind === "stage_started") {
        setActiveStage(line.stage);
      }
      if (line.stage && line.kind === "stage_completed") {
        setCompletedStages((previous) =>
          previous.includes(line.stage as PipelineStage) ? previous : [...previous, line.stage as PipelineStage],
        );
      }
      if (line.kind === "warning" && line.warning) {
        setActivity((previous) => [`Warning: ${line.warning}`, ...previous].slice(0, 18));
      }
      return;
    }

    if (line.type === "result") {
      if (line.result.finalStatus === "cancelled") {
        setScreen("setup");
        setResult(null);
        setErrorMessage("");
        setActivity([]);
        setActiveStage(null);
        setCompletedStages([]);
        return;
      }
      setResult(line.result);
      setScreen("success");
      setActiveStage(null);
      setErrorMessage("");
      return;
    }

    if (line.type === "error") {
      setErrorMessage(line.message);
      setScreen("error");
      return;
    }

    if (line.type === "exit" && line.code !== null) {
      if (line.code === 130) {
        setScreen("setup");
        setResult(null);
        setErrorMessage("");
        setActivity([]);
        setActiveStage(null);
        setCompletedStages([]);
        return;
      }
      if (line.code !== 0) {
        setErrorMessage("The worker exited before finishing.");
        setScreen("error");
      }
    }
  }

  function applyInputPath(path: string) {
    if (!path) {
      return;
    }

    setSettings((previous) => ({
      ...previous,
      inputPath: path,
      outputPath: deriveOutputName(path),
    }));
    setScreen((previous) => (previous === "processing" ? previous : "setup"));
    setErrorMessage("");
    setResult(null);
  }

  async function handlePickInput() {
    const path = await pickVideoFile();
    if (path) {
      applyInputPath(path);
    }
  }

  async function handlePickOutput() {
    const defaultName = settings.outputPath
      ? splitPath(settings.outputPath).base
      : settings.inputPath
        ? splitPath(deriveOutputName(settings.inputPath)).base
        : "clean_video.mp4";
    const defaultDir = settings.outputPath
      ? splitPath(settings.outputPath).dir
      : settings.inputPath
        ? splitPath(settings.inputPath).dir
        : undefined;
    const path = await pickOutputFile(defaultName, defaultDir || undefined);
    if (path) {
      setSettings((previous) => ({ ...previous, outputPath: path }));
    }
  }

  async function handleStart() {
    try {
      const request = buildGuiRequest(settings);
      setActivity(["Preparing worker..."]);
      setCompletedStages([]);
      setActiveStage(null);
      setResult(null);
      setErrorMessage("");
      setScreen("processing");
      await startPipelineJob(request);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Unable to start processing.");
      setScreen("error");
    }
  }

  async function handleCancel() {
    await cancelPipelineJob();
    setActivity((previous) => ["Cancellation requested...", ...previous].slice(0, 18));
  }

  async function handleReveal() {
    if (result?.outputPath) {
      await revealInFinder(result.outputPath);
    }
  }

  async function handleOpen() {
    if (result?.outputPath) {
      await openPath(result.outputPath);
    }
  }

  function resetFlow() {
    setSettings((previous) => ({
      ...previous,
      inputPath: "",
      outputPath: "",
      advancedOpen: false,
    }));
    setScreen("setup");
    setResult(null);
    setErrorMessage("");
    setActivity([]);
    setCompletedStages([]);
    setActiveStage(null);
    setIsDragActive(false);
  }

  return (
    <main className="app-shell" data-screen={screen}>
      <div className="shell-glow shell-glow-left" />
      <div className="shell-glow shell-glow-right" />

      <section className="app-window" data-view={screen}>
        {screen === "setup" ? (
          <SetupScreen
            settings={settings}
            selectedPreset={selectedPreset}
            resolvedModel={resolvedModel}
            resolvedExport={resolvedExport}
            canStart={canStart}
            errorMessage={errorMessage}
            isDragActive={isDragActive}
            onDragActiveChange={setIsDragActive}
            onPickInput={handlePickInput}
            onPickOutput={handlePickOutput}
            onPresetChange={(preset) => setSettings((previous) => applyPreset(previous, preset))}
            onToggleChange={(patch) => setSettings((previous) => ({ ...previous, ...patch }))}
            onStart={handleStart}
          />
        ) : null}

        {screen === "processing" ? (
          <ProcessingScreen
            activeStage={activeStage}
            completedStages={completedStages}
            activity={activity}
            outputPath={settings.outputPath}
            progress={processingProgress}
            onCancel={handleCancel}
          />
        ) : null}

        {screen === "success" && result ? (
          <SuccessScreen
            result={result}
            onOpen={handleOpen}
            onReveal={handleReveal}
            onReset={resetFlow}
          />
        ) : null}

        {screen === "error" ? (
          <ErrorScreen
            message={errorMessage || "The worker stopped unexpectedly."}
            onBack={resetFlow}
          />
        ) : null}
      </section>
    </main>
  );
}

function SetupScreen(props: {
  settings: UiSettings;
  selectedPreset: (typeof PRESETS)[PresetKey];
  resolvedModel: string;
  resolvedExport: string;
  canStart: boolean;
  errorMessage: string;
  isDragActive: boolean;
  onDragActiveChange: (active: boolean) => void;
  onPickInput: () => Promise<void>;
  onPickOutput: () => Promise<void>;
  onPresetChange: (preset: PresetKey) => void;
  onToggleChange: (patch: Partial<UiSettings>) => void;
  onStart: () => Promise<void>;
}) {
  const presetId = useId();
  const selectedFile = props.settings.inputPath ? splitPath(props.settings.inputPath).base : "";
  const outputBase = props.settings.outputPath ? splitPath(props.settings.outputPath).base : "Auto";

  function handleDragEnter(event: React.DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    props.onDragActiveChange(true);
  }

  function handleDragOver(event: React.DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    props.onDragActiveChange(true);
  }

  function handleDragLeave(event: React.DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    if (event.currentTarget.contains(event.relatedTarget as Node | null)) {
      return;
    }
    props.onDragActiveChange(false);
  }

  function handleDrop(event: React.DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    props.onDragActiveChange(false);
  }

  return (
    <div className="screen screen-setup">
      <header className="setup-header">
        <p className="app-title">UmmFiltered</p>
        <h1>Clean up talking-head videos</h1>
        <p>Remove filler words and smooth pauses. Instantly.</p>
      </header>

      <button
        className={
          props.isDragActive
            ? selectedFile
              ? "drop-zone drag-active selected"
              : "drop-zone drag-active"
            : selectedFile
              ? "drop-zone selected"
              : "drop-zone"
        }
        onClick={props.onPickInput}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        type="button"
      >
        <span className="drop-zone-backdrop" aria-hidden="true" />
        <div className="drop-zone-icon" aria-hidden="true">
          <VideoIcon />
        </div>
        <div className="drop-zone-copy">
          <strong>{selectedFile || "Drop your video here"}</strong>
          <span>{selectedFile ? "Drop another video or browse files" : "or browse files"}</span>
        </div>
        {selectedFile ? (
          <div className="drop-zone-meta">
            <span className="meta-pill">
              <b>Source</b>
              <em>{selectedFile}</em>
            </span>
            <span className="meta-pill subtle">
              <b>Output</b>
              <em>{outputBase}</em>
            </span>
          </div>
        ) : null}
      </button>

      <div className="action-bar">
        <div className="preset-control">
          <label className="control-label" htmlFor={presetId}>
            Preset:
          </label>
          <div className="select-shell">
            <span className="select-leading" aria-hidden="true">
              <PresetIcon />
            </span>
            <select
              id={presetId}
              value={props.settings.preset}
              onChange={(event) => props.onPresetChange(event.target.value as PresetKey)}
            >
              {(Object.keys(PRESETS) as PresetKey[]).map((preset) => (
                <option key={preset} value={preset}>
                  {PRESETS[preset].label}
                </option>
              ))}
            </select>
            <span className="select-icon" aria-hidden="true">
              <ChevronIcon />
            </span>
          </div>
        </div>

        <button className="primary-button" disabled={!props.canStart} onClick={props.onStart} type="button">
          <span>Clean My Video</span>
          <span className="button-arrow" aria-hidden="true">
            <ArrowIcon />
          </span>
        </button>
      </div>

      <div className="setup-footer">
        <div className="status-line">
          <span className="status-dot" />
          <span>Runs locally on your Mac - about 1 minute</span>
        </div>

        <button
          aria-label="More options"
          className={props.settings.advancedOpen ? "icon-button active" : "icon-button"}
          onClick={() => props.onToggleChange({ advancedOpen: !props.settings.advancedOpen })}
          type="button"
        >
          <GearIcon />
        </button>
      </div>

      {props.errorMessage ? <p className="inline-error">{props.errorMessage}</p> : null}

      {props.settings.advancedOpen ? (
        <section className="advanced-panel">
          <div className="advanced-panel-header">
            <div>
              <p className="advanced-kicker">More options</p>
              <strong>Adjust the cleanup pass</strong>
            </div>
            <button
              aria-label="Close options"
              className="icon-button subtle"
              onClick={() => props.onToggleChange({ advancedOpen: false })}
              type="button"
            >
              <CloseIcon />
            </button>
          </div>

          <p className="advanced-summary">
            {props.selectedPreset.label} defaults to a {props.resolvedModel} model with {props.resolvedExport} export.
          </p>

          <div className="toggle-list">
            <SwitchRow
              active={props.settings.aggressive}
              copy="Catch contextual fillers."
              label="Aggressive cleanup"
              onClick={() => props.onToggleChange({ aggressive: !props.settings.aggressive })}
            />
            <SwitchRow
              active={props.settings.verifyPass}
              copy="Run a verification pass after render."
              label="Verification pass"
              onClick={() => props.onToggleChange({ verifyPass: !props.settings.verifyPass })}
            />
            <SwitchRow
              active={props.settings.pauseMode !== "none"}
              copy="Keep natural pause smoothing enabled."
              label="Natural pauses"
              onClick={() =>
                props.onToggleChange({ pauseMode: props.settings.pauseMode === "none" ? "adaptive" : "none" })
              }
            />
          </div>

          <div className="advanced-grid">
            <Field label="Output">
              <button className="secondary-field-button" onClick={props.onPickOutput} type="button">
                {outputBase === "Auto" ? "Choose output file" : outputBase}
              </button>
            </Field>

            <Field label="Model size">
              <select
                value={props.settings.modelSizeOverride}
                onChange={(event) =>
                  props.onToggleChange({
                    modelSizeOverride: event.target.value as UiSettings["modelSizeOverride"],
                  })
                }
              >
                <option value="auto">Preset default</option>
                {["tiny", "base", "small", "medium", "large"].map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </Field>

            <Field label="Export quality">
              <select
                value={props.settings.qualityOverride}
                onChange={(event) =>
                  props.onToggleChange({
                    qualityOverride: event.target.value as UiSettings["qualityOverride"],
                  })
                }
              >
                <option value="auto">Preset default</option>
                <option value="matched">Matched</option>
                <option value="lossless">Lossless</option>
              </select>
            </Field>

            <Field label="Pause mode">
              <select
                value={props.settings.pauseMode}
                onChange={(event) =>
                  props.onToggleChange({
                    pauseMode: event.target.value as PauseMode,
                  })
                }
              >
                <option value="adaptive">Adaptive</option>
                <option value="none">None</option>
                <option value="custom">Custom ms</option>
              </select>
            </Field>

            <Field label="Custom pause (ms)">
              <input
                disabled={props.settings.pauseMode !== "custom"}
                max="1000"
                min="0"
                step="10"
                type="number"
                value={props.settings.customPauseMs}
                onChange={(event) => props.onToggleChange({ customPauseMs: event.target.value })}
              />
            </Field>

            <Field label="Minimum confidence">
              <input
                max="1"
                min="0"
                step="0.01"
                type="number"
                value={props.settings.minConfidence}
                onChange={(event) => props.onToggleChange({ minConfidence: event.target.value })}
              />
            </Field>

            <Field className="wide" label="Custom fillers">
              <input
                placeholder="anyway, right, okay"
                type="text"
                value={props.settings.customFillers}
                onChange={(event) => props.onToggleChange({ customFillers: event.target.value })}
              />
            </Field>
          </div>
        </section>
      ) : null}
    </div>
  );
}

function ProcessingScreen(props: {
  activeStage: PipelineStage | null;
  completedStages: PipelineStage[];
  activity: string[];
  outputPath: string;
  progress: number;
  onCancel: () => Promise<void>;
}) {
  return (
    <div className="screen state-screen">
      <div className="screen-progress" aria-hidden="true">
        <span style={{ width: `${Math.min(props.progress * 100, 100)}%` }} />
      </div>

      <header className="state-header">
        <p className="app-title muted">UmmFiltered</p>
        <h1>Cleaning your video</h1>
        <p>Running locally through transcription, filler detection, and render.</p>
      </header>

      <section className="processing-card">
        <div className="processing-stage">
          <span className="processing-label">Current stage</span>
          <strong>{props.activeStage ? STAGE_LABELS[props.activeStage] : "Preparing worker"}</strong>
          <small>{props.outputPath || "Output will be written next to the source clip."}</small>
        </div>

        <div className="stage-list">
          {STAGE_ORDER.map((stage) => {
            const done = props.completedStages.includes(stage);
            const active = props.activeStage === stage;
            return (
              <div key={stage} className={active ? "stage-chip active" : done ? "stage-chip done" : "stage-chip"}>
                <span className="stage-chip-dot" />
                <strong>{STAGE_LABELS[stage]}</strong>
              </div>
            );
          })}
        </div>
      </section>

      <section className="log-card">
        <div className="log-header">
          <span className="processing-label">Activity</span>
        </div>
        <div className="activity-log">
          {props.activity.length === 0 ? (
            <p className="activity-empty">Waiting for the worker to begin...</p>
          ) : (
            props.activity.map((entry, index) => (
              <div className="activity-row" key={`${entry}-${index}`}>
                {entry}
              </div>
            ))
          )}
        </div>
      </section>

      <footer className="state-footer">
        <button className="secondary-button" onClick={props.onCancel} type="button">
          Cancel
        </button>
      </footer>
    </div>
  );
}

function SuccessScreen(props: {
  result: PipelineResult;
  onOpen: () => Promise<void>;
  onReveal: () => Promise<void>;
  onReset: () => void;
}) {
  const ready = props.result.finalStatus !== "no_fillers";

  return (
    <div className="screen state-screen">
      <header className="state-header">
        <p className="app-title muted">UmmFiltered</p>
        <h1>{ready ? "Your video is ready" : "No cleanup needed"}</h1>
        <p>
          {ready
            ? "The cleaned export is ready to open or reveal in Finder."
            : "The source did not need any filler cleanup, so no new file was written."}
        </p>
      </header>

      <section className="result-grid">
        <MetricCard label="Removed fillers" value={String(props.result.removedFillers)} />
        <MetricCard label="Time removed" value={`${props.result.removedSeconds.toFixed(1)}s`} />
      </section>

      <section className="result-path">
        <span className="processing-label">{ready ? "Saved file" : "Status"}</span>
        <strong>{ready ? props.result.outputPath : "Already clean"}</strong>
      </section>

      <footer className="state-footer multi">
        {ready ? (
          <>
            <button className="secondary-button" onClick={props.onReveal} type="button">
              Reveal in Finder
            </button>
            <button className="secondary-button" onClick={props.onOpen} type="button">
              Open
            </button>
          </>
        ) : null}
        <button className="primary-button" onClick={props.onReset} type="button">
          Process Another
        </button>
      </footer>
    </div>
  );
}

function ErrorScreen(props: { message: string; onBack: () => void }) {
  return (
    <div className="screen state-screen">
      <header className="state-header">
        <p className="app-title muted">UmmFiltered</p>
        <h1>Could not finish this video</h1>
        <p>{props.message}</p>
      </header>

      <footer className="state-footer">
        <button className="primary-button" onClick={props.onBack} type="button">
          Back To Setup
        </button>
      </footer>
    </div>
  );
}

function Field(props: { label: string; children: ReactNode; className?: string }) {
  return (
    <label className={props.className ? `field ${props.className}` : "field"}>
      <span>{props.label}</span>
      {props.children}
    </label>
  );
}

function MetricCard(props: { label: string; value: string }) {
  return (
    <div className="metric-card">
      <small>{props.label}</small>
      <strong>{props.value}</strong>
    </div>
  );
}

function SwitchRow(props: {
  label: string;
  copy: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button className="switch-row" onClick={props.onClick} type="button">
      <div className="switch-copy">
        <strong>{props.label}</strong>
        <span>{props.copy}</span>
      </div>
      <span className={props.active ? "switch active" : "switch"}>
        <span />
      </span>
    </button>
  );
}

function VideoIcon() {
  return (
    <svg fill="none" height="34" viewBox="0 0 32 32" width="34" xmlns="http://www.w3.org/2000/svg">
      <rect height="18" rx="4" stroke="currentColor" strokeWidth="1.8" width="18" x="7" y="8" />
      <path d="M14 13.2L19 16L14 18.8V13.2Z" fill="currentColor" />
      <path d="M11 7H21" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
    </svg>
  );
}

function ChevronIcon() {
  return (
    <svg fill="none" height="12" viewBox="0 0 12 12" width="12" xmlns="http://www.w3.org/2000/svg">
      <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeLinecap="round" strokeWidth="1.5" />
    </svg>
  );
}

function ArrowIcon() {
  return (
    <svg fill="none" height="14" viewBox="0 0 14 14" width="14" xmlns="http://www.w3.org/2000/svg">
      <path d="M2.75 7H11.25" stroke="currentColor" strokeLinecap="round" strokeWidth="1.5" />
      <path d="M7.9 3.65L11.25 7L7.9 10.35" stroke="currentColor" strokeLinecap="round" strokeWidth="1.5" />
    </svg>
  );
}

function PresetIcon() {
  return (
    <svg fill="none" height="14" viewBox="0 0 14 14" width="14" xmlns="http://www.w3.org/2000/svg">
      <path d="M7 1.8L8.34 4.52L11.34 4.96L9.17 7.07L9.68 10.05L7 8.64L4.32 10.05L4.83 7.07L2.66 4.96L5.66 4.52L7 1.8Z" stroke="currentColor" strokeLinejoin="round" strokeWidth="1.2" />
    </svg>
  );
}

function GearIcon() {
  return (
    <svg fill="none" height="18" viewBox="0 0 20 20" width="18" xmlns="http://www.w3.org/2000/svg">
      <path
        d="M8.47 2.79C8.77 1.74 10.23 1.74 10.53 2.79L10.8 3.75C10.97 4.36 11.63 4.66 12.21 4.41L13.12 4.03C14.12 3.62 15.16 4.66 14.75 5.66L14.37 6.57C14.12 7.15 14.42 7.81 15.03 7.98L15.99 8.25C17.04 8.55 17.04 10.01 15.99 10.31L15.03 10.58C14.42 10.75 14.12 11.41 14.37 11.99L14.75 12.9C15.16 13.9 14.12 14.94 13.12 14.53L12.21 14.15C11.63 13.9 10.97 14.2 10.8 14.81L10.53 15.77C10.23 16.82 8.77 16.82 8.47 15.77L8.2 14.81C8.03 14.2 7.37 13.9 6.79 14.15L5.88 14.53C4.88 14.94 3.84 13.9 4.25 12.9L4.63 11.99C4.88 11.41 4.58 10.75 3.97 10.58L3.01 10.31C1.96 10.01 1.96 8.55 3.01 8.25L3.97 7.98C4.58 7.81 4.88 7.15 4.63 6.57L4.25 5.66C3.84 4.66 4.88 3.62 5.88 4.03L6.79 4.41C7.37 4.66 8.03 4.36 8.2 3.75L8.47 2.79Z"
        stroke="currentColor"
        strokeWidth="1.3"
      />
      <circle cx="10" cy="9.28" r="2.35" stroke="currentColor" strokeWidth="1.3" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg fill="none" height="14" viewBox="0 0 14 14" width="14" xmlns="http://www.w3.org/2000/svg">
      <path d="M3.5 3.5L10.5 10.5" stroke="currentColor" strokeLinecap="round" strokeWidth="1.5" />
      <path d="M10.5 3.5L3.5 10.5" stroke="currentColor" strokeLinecap="round" strokeWidth="1.5" />
    </svg>
  );
}

function formatActivity(line: Extract<WorkerLine, { type: "event" }>): string {
  const prefix = line.stage ? `${STAGE_LABELS[line.stage]}:` : "System:";
  return `${prefix} ${line.warning || line.message}`;
}
