import type {
  GuiProcessRequest,
  ModelSize,
  PauseMode,
  PresetKey,
  QualityMode,
  UiSettings,
} from "./types";

export const SETTINGS_STORAGE_KEY = "ummfiltered.desktop.settings.v1";

type PersistedSettings = Pick<
  UiSettings,
  | "preset"
  | "aggressive"
  | "verifyPass"
  | "pauseMode"
  | "customPauseMs"
  | "modelSizeOverride"
  | "qualityOverride"
  | "minConfidence"
  | "customFillers"
>;

const PERSISTED_SETTING_KEYS: Array<keyof PersistedSettings> = [
  "preset",
  "aggressive",
  "verifyPass",
  "pauseMode",
  "customPauseMs",
  "modelSizeOverride",
  "qualityOverride",
  "minConfidence",
  "customFillers",
];

export const PRESETS: Record<
  PresetKey,
  {
    label: string;
    tagline: string;
    modelSize: ModelSize;
    quality: QualityMode;
    verifyPass: boolean;
  }
> = {
  speed: {
    label: "Conservative",
    tagline: "Fastest pass with gentler cleanup.",
    modelSize: "base",
    quality: "matched",
    verifyPass: false,
  },
  balanced: {
    label: "Natural Cleanup",
    tagline: "Default cleanup for everyday talking-head edits.",
    modelSize: "medium",
    quality: "matched",
    verifyPass: true,
  },
  quality: {
    label: "High Precision",
    tagline: "Most thorough local pass for difficult audio.",
    modelSize: "large",
    quality: "lossless",
    verifyPass: true,
  },
};

export const STAGE_ORDER = [
  "probe",
  "extract_audio",
  "transcribe",
  "detect_fillers",
  "plan_cuts",
  "render",
  "verify",
  "final_check",
] as const;

export function createDefaultSettings(): UiSettings {
  return {
    inputPath: "",
    outputPath: "",
    preset: "balanced",
    aggressive: false,
    verifyPass: PRESETS.balanced.verifyPass,
    pauseMode: "adaptive",
    customPauseMs: "120",
    modelSizeOverride: "auto",
    qualityOverride: "auto",
    minConfidence: "0.15",
    customFillers: "",
    advancedOpen: false,
  };
}

export function applyPreset(settings: UiSettings, preset: PresetKey): UiSettings {
  return {
    ...settings,
    preset,
    verifyPass: PRESETS[preset].verifyPass,
    modelSizeOverride: "auto",
    qualityOverride: "auto",
  };
}

export function buildGuiRequest(settings: UiSettings): GuiProcessRequest {
  if (!settings.inputPath) {
    throw new Error("Choose a video before processing.");
  }

  const fixedPauseMs =
    settings.pauseMode === "custom" ? parseFiniteNumber(settings.customPauseMs, "Enter a valid custom pause.") : undefined;

  const minConfidence = parseFiniteNumber(settings.minConfidence, "Enter a valid minimum confidence.");

  return {
    inputPath: settings.inputPath,
    outputPath: settings.outputPath || undefined,
    preset: settings.preset,
    aggressive: settings.aggressive,
    verifyPass: settings.verifyPass,
    naturalPauses: settings.pauseMode === "adaptive",
    overrides: {
      modelSize: settings.modelSizeOverride === "auto" ? undefined : settings.modelSizeOverride,
      quality: settings.qualityOverride === "auto" ? undefined : settings.qualityOverride,
      minConfidence,
      customFillers: parseCustomFillers(settings.customFillers),
      fixedPauseMs,
    },
  };
}

export function loadPersistedSettings(): UiSettings {
  if (typeof window === "undefined") {
    return createDefaultSettings();
  }

  const raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
  if (!raw) {
    return createDefaultSettings();
  }

  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const restored: Partial<PersistedSettings> = {};

    for (const key of PERSISTED_SETTING_KEYS) {
      const value = parsed[key];
      if (value !== undefined) {
        (restored as Record<string, unknown>)[key] = value;
      }
    }

    return { ...createDefaultSettings(), ...restored };
  } catch {
    return createDefaultSettings();
  }
}

export function persistSettings(settings: UiSettings): void {
  if (typeof window === "undefined") {
    return;
  }
  const persisted: PersistedSettings = {
    preset: settings.preset,
    aggressive: settings.aggressive,
    verifyPass: settings.verifyPass,
    pauseMode: settings.pauseMode,
    customPauseMs: settings.customPauseMs,
    modelSizeOverride: settings.modelSizeOverride,
    qualityOverride: settings.qualityOverride,
    minConfidence: settings.minConfidence,
    customFillers: settings.customFillers,
  };
  window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(persisted));
}

export function resolvedModelSize(settings: UiSettings): ModelSize {
  return settings.modelSizeOverride === "auto"
    ? PRESETS[settings.preset].modelSize
    : settings.modelSizeOverride;
}

export function resolvedQuality(settings: UiSettings): QualityMode {
  return settings.qualityOverride === "auto"
    ? PRESETS[settings.preset].quality
    : settings.qualityOverride;
}

export function deriveOutputName(inputPath: string): string {
  const { dir, name, ext } = splitPath(inputPath);
  return joinPath(dir, `${name}_ummfiltered${ext}`);
}

export function splitPath(path: string): { dir: string; name: string; ext: string; base: string } {
  const normalized = path.replace(/\\/g, "/");
  const parts = normalized.split("/");
  const base = parts[parts.length - 1] ?? "";
  const dot = base.lastIndexOf(".");
  const name = dot > 0 ? base.slice(0, dot) : base;
  const ext = dot > 0 ? base.slice(dot) : "";
  return {
    dir: parts.slice(0, -1).join("/"),
    name,
    ext,
    base,
  };
}

export function joinPath(dir: string, base: string): string {
  if (!dir) {
    return base;
  }
  return `${dir.replace(/\/$/, "")}/${base}`;
}

export function pauseModeLabel(pauseMode: PauseMode): string {
  if (pauseMode === "adaptive") {
    return "Adaptive";
  }
  if (pauseMode === "none") {
    return "None";
  }
  return "Custom";
}

function parseCustomFillers(value: string): string[] | undefined {
  const normalized = value
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
  return normalized.length > 0 ? normalized : undefined;
}

function parseFiniteNumber(value: string, errorMessage: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(errorMessage);
  }
  return parsed;
}
