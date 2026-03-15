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

  it("restores persisted settings without skipping the wizard", async () => {
    window.localStorage.setItem(
      "ummfiltered.desktop.settings.v1",
      JSON.stringify({
        inputPath: "/tmp/restored.mov",
        preset: "quality",
        aggressive: true,
      }),
    );

    render(<App />);

    expect(screen.getByRole("heading", { name: /welcome to ummfiltered/i })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /start setup/i }));
    await userEvent.click(screen.getByRole("button", { name: /choose a talking-head clip/i }));
    await userEvent.click(screen.getByRole("button", { name: /^continue$/i }));
    await userEvent.click(screen.getByRole("button", { name: /^continue$/i }));

    expect(screen.getByRole("heading", { name: /review your cleanup settings/i })).toBeInTheDocument();
    expect(screen.getByText("Quality")).toBeInTheDocument();
    expect(screen.getByText(/large model · lossless export · adaptive/i)).toBeInTheDocument();
  });

  it("moves through the wizard and finishes successfully", async () => {
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /start setup/i }));

    expect(screen.getByRole("heading", { name: /choose your source clip/i })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /choose a talking-head clip/i }));

    expect(await screen.findByTestId("input-path")).toHaveTextContent("/tmp/demo.mov");

    await userEvent.click(screen.getByRole("button", { name: /^continue$/i }));

    expect(screen.getByRole("heading", { name: /pick your cleanup profile/i })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /^continue$/i }));

    expect(screen.getByRole("heading", { name: /review your cleanup settings/i })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /start cleanup/i }));

    expect(tauriMocks.startPipelineJob).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("heading", { name: /cleaning up your video/i })).toBeInTheDocument();

    await act(async () => {
      workerHandler?.({
        type: "event",
        kind: "stage_started",
        stage: "render",
        message: "Stitching it all together...",
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

    expect(screen.getByRole("heading", { name: /render complete/i })).toBeInTheDocument();
    expect(screen.getByText("/tmp/demo_ummfiltered.mov")).toBeInTheDocument();
    expect(screen.getByText("7")).toBeInTheDocument();
  });
});
