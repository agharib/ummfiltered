import {
  SETTINGS_STORAGE_KEY,
  applyPreset,
  buildGuiRequest,
  createDefaultSettings,
  loadPersistedSettings,
  persistSettings,
} from "./settings";

describe("settings helpers", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("applies preset defaults and resets model and quality overrides", () => {
    const initial = {
      ...createDefaultSettings(),
      verifyPass: true,
      modelSizeOverride: "tiny" as const,
      qualityOverride: "lossless" as const,
    };

    const next = applyPreset(initial, "speed");

    expect(next.verifyPass).toBe(false);
    expect(next.modelSizeOverride).toBe("auto");
    expect(next.qualityOverride).toBe("auto");
  });

  it("builds a worker request with advanced overrides", () => {
    const settings = {
      ...createDefaultSettings(),
      inputPath: "/tmp/demo.mov",
      outputPath: "/tmp/demo_ummfiltered.mov",
      preset: "quality" as const,
      aggressive: true,
      verifyPass: false,
      pauseMode: "custom" as const,
      customPauseMs: "140",
      modelSizeOverride: "small" as const,
      qualityOverride: "matched" as const,
      minConfidence: "0.32",
      customFillers: "anyway, right",
    };

    expect(buildGuiRequest(settings)).toEqual({
      inputPath: "/tmp/demo.mov",
      outputPath: "/tmp/demo_ummfiltered.mov",
      preset: "quality",
      aggressive: true,
      verifyPass: false,
      naturalPauses: false,
      overrides: {
        modelSize: "small",
        quality: "matched",
        minConfidence: 0.32,
        customFillers: ["anyway", "right"],
        fixedPauseMs: 140,
      },
    });
  });

  it("persists and restores settings from local storage", () => {
    const settings = {
      ...createDefaultSettings(),
      inputPath: "/tmp/demo.mov",
      outputPath: "/tmp/demo_ummfiltered.mov",
      customFillers: "like, so",
      advancedOpen: true,
    };

    persistSettings(settings);

    expect(window.localStorage.getItem(SETTINGS_STORAGE_KEY)).not.toContain("/tmp/demo.mov");
    expect(loadPersistedSettings()).toMatchObject({
      inputPath: "",
      outputPath: "",
      customFillers: "like, so",
      advancedOpen: false,
    });
  });
});
