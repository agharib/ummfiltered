import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type { WorkerLine } from "./lib/types";

const tauriMocks = vi.hoisted(() => ({
  startPipelineJob: vi.fn().mockResolvedValue(undefined),
}));

let workerHandler: ((payload: WorkerLine) => void) | undefined;

vi.mock("./lib/tauri", () => ({
  startPipelineJob: tauriMocks.startPipelineJob,
  cancelPipelineJob: vi.fn().mockResolvedValue(undefined),
  pickVideoFile: vi.fn().mockResolvedValue("/tmp/demo.mov"),
  pickOutputFile: vi.fn().mockResolvedValue("/tmp/demo_ummfiltered.mov"),
  openPath: vi.fn().mockResolvedValue(undefined),
  revealInFinder: vi.fn().mockResolvedValue(undefined),
  subscribeToPipelineMessages: vi.fn(async (handler: (payload: WorkerLine) => void) => {
    workerHandler = handler;
    return () => {};
  }),
  subscribeToFileDrops: vi.fn(async () => () => {}),
}));

describe("App", () => {
  beforeEach(() => {
    window.localStorage.clear();
    workerHandler = undefined;
    tauriMocks.startPipelineJob.mockClear();
  });

  it("restores persisted settings while staying on the setup screen", () => {
    window.localStorage.setItem(
      "ummfiltered.desktop.settings.v1",
      JSON.stringify({
        preset: "quality",
        aggressive: true,
      }),
    );

    render(<App />);

    expect(screen.getByRole("heading", { name: /clean up talking-head videos/i })).toBeInTheDocument();
    expect(screen.getByLabelText(/preset:/i)).toHaveValue("quality");
    expect(screen.getByRole("button", { name: /clean my video/i })).toBeDisabled();
  });

  it("lets you choose a file and complete the processing flow", async () => {
    render(<App />);

    const pickButton = screen.getByRole("button", { name: /drop your video here/i });
    const startButton = screen.getByRole("button", { name: /clean my video/i });

    expect(startButton).toBeDisabled();

    await userEvent.click(pickButton);

    expect(await screen.findByRole("button", { name: /demo\.mov/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /clean my video/i })).toBeEnabled();

    await userEvent.click(screen.getByRole("button", { name: /clean my video/i }));

    expect(tauriMocks.startPipelineJob).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("heading", { name: /cleaning your video/i })).toBeInTheDocument();

    await act(async () => {
      workerHandler?.({
        type: "event",
        kind: "stage_started",
        stage: "render",
        message: "Rendering output",
      });
      workerHandler?.({
        type: "result",
        result: {
          outputPath: "/tmp/demo_ummfiltered.mov",
          removedFillers: 7,
          removedSeconds: 1.9,
          warnings: [],
          finalStatus: "success",
        },
      });
    });

    expect(screen.getByRole("heading", { name: /your video is ready/i })).toBeInTheDocument();
    expect(screen.getByText("/tmp/demo_ummfiltered.mov")).toBeInTheDocument();
    expect(screen.getByText("7")).toBeInTheDocument();
  });
});
