import { type ReactNode, useEffect, useState } from "react";

import {
  PRESETS,
  STAGE_ORDER,
  applyPreset,
  buildGuiRequest,
  deriveOutputName,
  loadPersistedSettings,
  pauseModeLabel,
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
  UiSettings,
  WorkerLine,
} from "./lib/types";

type ScreenState = "setup" | "processing" | "success" | "error";
type SetupStep = "welcome" | "input" | "preset" | "review";

const SETUP_STEPS: SetupStep[] = ["welcome", "input", "preset", "review"];

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

const STEP_TITLES: Record<SetupStep, string> = {
  welcome: "Welcome to UmmFiltered",
  input: "Choose your source clip",
  preset: "Pick your cleanup profile",
  review: "Review your cleanup settings",
};

const STEP_DESCRIPTIONS: Record<SetupStep, string> = {
  welcome: "We’ll guide you through a short setup and have your first talking-head cleanup ready in under a minute.",
  input: "Drag a video onto the window or choose one from disk. We’ll keep the output next to the source unless you override it.",
  preset: "Start with the profile that matches your goal. You can still fine-tune a few switches before rendering.",
  review: "Keep the final screen compact: the important toggles stay up front and everything deeper folds underneath.",
};

export default function App() {
  const [settings, setSettings] = useState<UiSettings>(() => loadPersistedSettings());
  const [setupStep, setSetupStep] = useState<SetupStep>("welcome");
  const [screen, setScreen] = useState<ScreenState>("setup");
  const [activity, setActivity] = useState<string[]>([]);
  const [activeStage, setActiveStage] = useState<PipelineStage | null>(null);
  const [completedStages, setCompletedStages] = useState<PipelineStage[]>([]);
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [errorMessage, setErrorMessage] = useState("");

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
      applyInputPath(path);
    }).then((dispose) => {
      offDrops = dispose;
    });

    return () => {
      offMessages?.();
      offDrops?.();
    };
  }, []);

  const setupIndex = SETUP_STEPS.indexOf(setupStep);
  const pipelineIndex = activeStage ? STAGE_ORDER.indexOf(activeStage) + 1 : completedStages.length;
  const progress =
    screen === "setup"
      ? (setupIndex + 1) / SETUP_STEPS.length
      : screen === "processing"
        ? Math.max(0.1, pipelineIndex / STAGE_ORDER.length)
        : 1;

  const resolvedModel = resolvedModelSize(settings);
  const resolvedExport = resolvedQuality(settings);
  const selectedPreset = PRESETS[settings.preset];
  const canContinueInput = Boolean(settings.inputPath);
  const canStart = Boolean(settings.inputPath) && screen !== "processing";

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
        setErrorMessage("Processing cancelled.");
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
        setErrorMessage("Processing cancelled.");
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
    setSetupStep((previous) => (previous === "welcome" ? "input" : previous));
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
      setActivity(["Preparing worker…"]);
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
    setActivity((previous) => ["Cancellation requested…", ...previous].slice(0, 18));
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
    setSetupStep("welcome");
    setResult(null);
    setErrorMessage("");
    setActivity([]);
    setCompletedStages([]);
    setActiveStage(null);
  }

  function advanceSetup() {
    if (setupStep === "welcome") {
      setSetupStep("input");
      return;
    }
    if (setupStep === "input" && canContinueInput) {
      setSetupStep("preset");
      return;
    }
    if (setupStep === "preset") {
      setSetupStep("review");
    }
  }

  function retreatSetup() {
    const current = SETUP_STEPS.indexOf(setupStep);
    if (current > 0) {
      setSetupStep(SETUP_STEPS[current - 1] ?? "welcome");
    }
  }

  return (
    <main className="wizard-shell" data-screen={screen}>
      <div className="wizard-glow wizard-glow-left" />
      <div className="wizard-glow wizard-glow-right" />
      <section className="wizard-card" data-view={screen === "setup" ? setupStep : screen}>
        <div className="wizard-ribbons" />
        <div className="wizard-progress">
          <span style={{ width: `${Math.min(progress * 100, 100)}%` }} />
        </div>

        {screen === "setup" ? (
          <SetupScreen
            setupStep={setupStep}
            settings={settings}
            selectedPreset={selectedPreset}
            resolvedModel={resolvedModel}
            resolvedExport={resolvedExport}
            canContinueInput={canContinueInput}
            canStart={canStart}
            errorMessage={errorMessage}
            onAdvance={advanceSetup}
            onBack={retreatSetup}
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
  setupStep: SetupStep;
  settings: UiSettings;
  selectedPreset: (typeof PRESETS)[keyof typeof PRESETS];
  resolvedModel: string;
  resolvedExport: string;
  canContinueInput: boolean;
  canStart: boolean;
  errorMessage: string;
  onAdvance: () => void;
  onBack: () => void;
  onPickInput: () => Promise<void>;
  onPickOutput: () => Promise<void>;
  onPresetChange: (preset: keyof typeof PRESETS) => void;
  onToggleChange: (patch: Partial<UiSettings>) => void;
  onStart: () => Promise<void>;
}) {
  const { setupStep, settings } = props;

  return (
    <div className={setupStep === "welcome" ? "wizard-step welcome-step" : "wizard-step"}>
      <header className="step-header">
        <div className="step-copy">
          <p className="step-kicker">
            Step {SETUP_STEPS.indexOf(setupStep) + 1} of {SETUP_STEPS.length}
          </p>
          <h1>{STEP_TITLES[setupStep]}</h1>
          <p>{STEP_DESCRIPTIONS[setupStep]}</p>
        </div>
        {setupStep !== "welcome" ? (
          <button className="step-back" onClick={props.onBack} type="button">
            Back
          </button>
        ) : null}
      </header>

      <div className="step-body">
        {setupStep === "welcome" ? (
          <div className="welcome-hero">
            <div className="welcome-pills" aria-label="Welcome highlights">
              <span>Local-only</span>
              <span>3 quick decisions</span>
              <span>Under 1 minute</span>
            </div>

            <div className="welcome-summary">
              <p className="welcome-caption">Clean up talking-head videos without opening a timeline.</p>
              <strong>Pick a clip, choose a preset, and let UmmFiltered make the first polished pass for you.</strong>
              <p>No cloud handoff, no review dashboard, and no busy setup screen before you can get to work.</p>
            </div>
          </div>
        ) : null}

        {setupStep === "input" ? (
          <div className="stack">
            <button className="hero-select" onClick={props.onPickInput} type="button">
              <span className="hero-select-kicker">Drag video here or browse</span>
              <strong>{settings.inputPath || "Choose a talking-head clip"}</strong>
              <small>We’ll use the source folder for output unless you change it.</small>
            </button>

            <div className="info-block" data-testid="input-path">
              <span className="info-label">Source clip</span>
              <strong>{settings.inputPath || "No file selected yet"}</strong>
            </div>

            <div className="info-block">
              <div className="info-head">
                <div>
                  <span className="info-label">Output location</span>
                  <strong>{settings.outputPath || "Auto: same folder with _ummfiltered suffix"}</strong>
                </div>
                <button className="ghost-pill" onClick={props.onPickOutput} type="button">
                  Change
                </button>
              </div>
            </div>
          </div>
        ) : null}

        {setupStep === "preset" ? (
          <div className="choice-grid">
            {(Object.keys(PRESETS) as Array<keyof typeof PRESETS>).map((preset) => (
              <button
                key={preset}
                className={preset === settings.preset ? "choice-card active" : "choice-card"}
                onClick={() => props.onPresetChange(preset)}
                type="button"
              >
                <div className="choice-icon">{preset === "speed" ? "S" : preset === "balanced" ? "B" : "Q"}</div>
                <strong>{PRESETS[preset].label}</strong>
                <small>{PRESETS[preset].tagline}</small>
                <span className="choice-meta">
                  {PRESETS[preset].modelSize} · {PRESETS[preset].quality}
                </span>
              </button>
            ))}
          </div>
        ) : null}

        {setupStep === "review" ? (
          <div className="stack">
            <div className="review-summary">
              <div>
                <span className="info-label">Selected preset</span>
                <strong>{props.selectedPreset.label}</strong>
                <small>
                  {props.resolvedModel} model · {props.resolvedExport} export · {pauseModeLabel(settings.pauseMode)}
                </small>
              </div>
              <div>
                <span className="info-label">Source</span>
                <strong>{splitPath(settings.inputPath || "No clip selected").base}</strong>
              </div>
            </div>

            <div className="choice-grid compact">
              <ToggleCard
                title="Aggressive cleanup"
                copy='Catch contextual fillers like "like" and "basically".'
                active={settings.aggressive}
                onClick={() => props.onToggleChange({ aggressive: !settings.aggressive })}
              />
              <ToggleCard
                title="Verification pass"
                copy="Recheck the render and rerender when the transcript looks worse."
                active={settings.verifyPass}
                onClick={() => props.onToggleChange({ verifyPass: !settings.verifyPass })}
              />
              <ToggleCard
                title="Natural pauses"
                copy="Keep adaptive micro-pauses so cuts still feel like speech."
                active={settings.pauseMode !== "none"}
                onClick={() =>
                  props.onToggleChange({ pauseMode: settings.pauseMode === "none" ? "adaptive" : "none" })
                }
              />
            </div>

            <details className="review-advanced" open={settings.advancedOpen}>
              <summary onClick={() => props.onToggleChange({ advancedOpen: !settings.advancedOpen })}>
                More options
              </summary>
              <div className="advanced-stack">
                <Field label="Model size override">
                  <select
                    value={settings.modelSizeOverride}
                    onChange={(event) =>
                      props.onToggleChange({
                        modelSizeOverride: event.target.value as UiSettings["modelSizeOverride"],
                      })
                    }
                  >
                    <option value="auto">Preset default ({PRESETS[settings.preset].modelSize})</option>
                    {["tiny", "base", "small", "medium", "large"].map((value) => (
                      <option key={value} value={value}>
                        {value}
                      </option>
                    ))}
                  </select>
                </Field>

                <Field label="Export quality override">
                  <select
                    value={settings.qualityOverride}
                    onChange={(event) =>
                      props.onToggleChange({
                        qualityOverride: event.target.value as UiSettings["qualityOverride"],
                      })
                    }
                  >
                    <option value="auto">Preset default ({PRESETS[settings.preset].quality})</option>
                    <option value="matched">Matched</option>
                    <option value="lossless">Lossless</option>
                  </select>
                </Field>

                <Field label="Minimum confidence">
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    value={settings.minConfidence}
                    onChange={(event) => props.onToggleChange({ minConfidence: event.target.value })}
                  />
                </Field>

                <Field label="Pause mode">
                  <select
                    value={settings.pauseMode}
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
                    type="number"
                    step="10"
                    min="0"
                    value={settings.customPauseMs}
                    disabled={settings.pauseMode !== "custom"}
                    onChange={(event) => props.onToggleChange({ customPauseMs: event.target.value })}
                  />
                </Field>

                <Field label="Custom fillers">
                  <input
                    type="text"
                    placeholder="anyway, right, okay"
                    value={settings.customFillers}
                    onChange={(event) => props.onToggleChange({ customFillers: event.target.value })}
                  />
                </Field>
              </div>
            </details>
          </div>
        ) : null}
      </div>

      {props.errorMessage ? <p className="inline-error">{props.errorMessage}</p> : null}

      <footer className="step-footer">
        {setupStep === "welcome" ? (
          <button className="primary-cta" onClick={props.onAdvance} type="button">
            Start setup
          </button>
        ) : null}
        {setupStep === "input" ? (
          <>
            <button className="secondary-cta" onClick={props.onBack} type="button">
              Back
            </button>
            <button
              className="primary-cta"
              disabled={!props.canContinueInput}
              onClick={props.onAdvance}
              type="button"
            >
              Continue
            </button>
          </>
        ) : null}
        {setupStep === "preset" ? (
          <>
            <button className="secondary-cta" onClick={props.onBack} type="button">
              Back
            </button>
            <button className="primary-cta" onClick={props.onAdvance} type="button">
              Continue
            </button>
          </>
        ) : null}
        {setupStep === "review" ? (
          <>
            <button className="secondary-cta" onClick={props.onBack} type="button">
              Back
            </button>
            <button className="primary-cta" disabled={!props.canStart} onClick={props.onStart} type="button">
              Start cleanup
            </button>
          </>
        ) : null}
      </footer>
    </div>
  );
}

function ProcessingScreen(props: {
  activeStage: PipelineStage | null;
  completedStages: PipelineStage[];
  activity: string[];
  outputPath: string;
  onCancel: () => Promise<void>;
}) {
  return (
    <div className="wizard-step">
      <header className="step-header">
        <div className="step-copy">
          <p className="step-kicker">Processing</p>
          <h1>Cleaning up your video</h1>
          <p>
            The Python pipeline is running underneath this wizard. You can watch each stage move forward without the UI turning into a dashboard.
          </p>
        </div>
      </header>

      <div className="step-body stack">
        <div className="stage-chip-grid">
          {STAGE_ORDER.map((stage) => {
            const done = props.completedStages.includes(stage);
            const active = props.activeStage === stage;
            return (
              <div key={stage} className={active ? "stage-chip active" : done ? "stage-chip done" : "stage-chip"}>
                <span />
                <strong>{STAGE_LABELS[stage]}</strong>
              </div>
            );
          })}
        </div>

        <div className="processing-panel">
          <div className="processing-copy">
            <span className="info-label">Current stage</span>
            <strong>{props.activeStage ? STAGE_LABELS[props.activeStage] : "Preparing worker"}</strong>
            <small>{props.outputPath || "Output will be created next to the source clip."}</small>
          </div>
          <div className="activity-log">
            {props.activity.length === 0 ? (
              <p className="activity-empty">Waiting for the worker to begin…</p>
            ) : (
              props.activity.map((entry, index) => (
                <div className="activity-row" key={`${entry}-${index}`}>
                  {entry}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <footer className="step-footer">
        <button className="secondary-cta" onClick={props.onCancel} type="button">
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
  return (
    <div className="wizard-step">
      <header className="step-header">
        <div className="step-copy">
          <p className="step-kicker">Complete</p>
          <h1>{props.result.finalStatus === "no_fillers" ? "Already clean" : "Render complete"}</h1>
          <p>
            {props.result.finalStatus === "no_fillers"
              ? "The source didn’t need a cleanup pass, so we stopped early."
              : "Your export is ready. Open it, reveal it in Finder, or jump straight into another clip."}
          </p>
        </div>
      </header>

      <div className="step-body stack">
        <div className="metric-strip">
          <MetricCard label="Removed fillers" value={String(props.result.removedFillers)} />
          <MetricCard label="Dead air removed" value={`${props.result.removedSeconds.toFixed(1)}s`} />
        </div>
        <div className="info-block">
          <span className="info-label">{props.result.finalStatus === "no_fillers" ? "Export" : "Saved file"}</span>
          <strong>
            {props.result.finalStatus === "no_fillers"
              ? "No new file was written because the source was already clean."
              : props.result.outputPath}
          </strong>
        </div>
      </div>

      <footer className="step-footer">
        {props.result.finalStatus !== "no_fillers" ? (
          <>
            <button className="secondary-cta" onClick={props.onReveal} type="button">
              Reveal in Finder
            </button>
            <button className="secondary-cta" onClick={props.onOpen} type="button">
              Open
            </button>
          </>
        ) : null}
        <button className="primary-cta" onClick={props.onReset} type="button">
          Process another
        </button>
      </footer>
    </div>
  );
}

function ErrorScreen(props: { message: string; onBack: () => void }) {
  return (
    <div className="wizard-step">
      <header className="step-header">
        <div className="step-copy">
          <p className="step-kicker">Attention</p>
          <h1>Something needs a quick fix</h1>
          <p>{props.message}</p>
        </div>
      </header>

      <footer className="step-footer">
        <button className="primary-cta" onClick={props.onBack} type="button">
          Back to setup
        </button>
      </footer>
    </div>
  );
}

function formatActivity(line: Extract<WorkerLine, { type: "event" }>): string {
  const prefix = line.stage ? `${STAGE_LABELS[line.stage]}:` : "System:";
  return `${prefix} ${line.warning || line.message}`;
}

function Field(props: { label: string; children: ReactNode }) {
  return (
    <label className="field">
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

function ToggleCard(props: {
  title: string;
  copy: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button className={props.active ? "toggle-card active" : "toggle-card"} onClick={props.onClick} type="button">
      <span className="toggle-dot" />
      <strong>{props.title}</strong>
      <small>{props.copy}</small>
    </button>
  );
}
