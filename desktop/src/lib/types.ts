export type PipelineStage =
  | "probe"
  | "extract_audio"
  | "transcribe"
  | "detect_fillers"
  | "plan_cuts"
  | "render"
  | "verify"
  | "final_check";

export type PipelineEventKind =
  | "stage_started"
  | "stage_completed"
  | "info"
  | "warning"
  | "error"
  | "cancelled";

export type PresetKey = "speed" | "balanced" | "quality";
export type ModelSize = "tiny" | "base" | "small" | "medium" | "large";
export type QualityMode = "matched" | "lossless";
export type PauseMode = "adaptive" | "none" | "custom";

export interface GuiOverrides {
  modelSize?: ModelSize;
  quality?: QualityMode;
  minConfidence?: number;
  customFillers?: string[];
  fixedPauseMs?: number;
}

export interface GuiProcessRequest {
  inputPath: string;
  outputPath?: string;
  preset: PresetKey;
  aggressive: boolean;
  verifyPass: boolean;
  naturalPauses: boolean;
  overrides: GuiOverrides;
}

export interface PipelineResult {
  outputPath: string;
  removedFillers: number;
  removedSeconds: number;
  warnings: string[];
  finalStatus: "success" | "no_fillers" | "dry_run" | "cancelled" | "error";
}

export interface WorkerEventLine {
  type: "event";
  kind: PipelineEventKind;
  stage: PipelineStage | null;
  message: string;
  warning?: string | null;
  stats?: Record<string, unknown> | null;
}

export interface WorkerResultLine {
  type: "result";
  result: PipelineResult;
}

export interface WorkerErrorLine {
  type: "error";
  code: string;
  message: string;
  details?: string;
}

export interface WorkerExitLine {
  type: "exit";
  code: number | null;
}

export type WorkerLine =
  | WorkerEventLine
  | WorkerResultLine
  | WorkerErrorLine
  | WorkerExitLine;

export interface UiSettings {
  inputPath: string;
  outputPath: string;
  preset: PresetKey;
  aggressive: boolean;
  verifyPass: boolean;
  pauseMode: PauseMode;
  customPauseMs: string;
  modelSizeOverride: "auto" | ModelSize;
  qualityOverride: "auto" | QualityMode;
  minConfidence: string;
  customFillers: string;
  advancedOpen: boolean;
}
