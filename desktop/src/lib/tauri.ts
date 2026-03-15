import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";

import type { GuiProcessRequest, WorkerLine } from "./types";

export async function startPipelineJob(request: GuiProcessRequest): Promise<void> {
  if (!isTauriRuntime()) {
    throw new Error("The desktop runtime is not available.");
  }
  await invoke("start_pipeline_job", { request });
}

export async function cancelPipelineJob(): Promise<void> {
  if (!isTauriRuntime()) {
    return;
  }
  await invoke("cancel_pipeline_job");
}

export async function pickVideoFile(): Promise<string | null> {
  if (!isTauriRuntime()) {
    return null;
  }
  return invoke<string | null>("pick_input_file");
}

export async function pickOutputFile(
  defaultName: string,
  defaultDir?: string,
): Promise<string | null> {
  if (!isTauriRuntime()) {
    return null;
  }
  return invoke<string | null>("pick_output_file", { defaultName, defaultDir });
}

export async function openPath(path: string): Promise<void> {
  if (!isTauriRuntime()) {
    return;
  }
  await invoke("open_path", { path });
}

export async function revealInFinder(path: string): Promise<void> {
  if (!isTauriRuntime()) {
    return;
  }
  await invoke("reveal_in_finder", { path });
}

export async function subscribeToPipelineMessages(
  onMessage: (payload: WorkerLine) => void,
): Promise<() => void> {
  if (!isTauriRuntime()) {
    return () => {};
  }
  const offMessage = await listen<WorkerLine>("pipeline://message", (event) => {
    onMessage(event.payload);
  });
  const offExit = await listen<{ code: number | null }>("pipeline://exit", (event) => {
    onMessage({ type: "exit", code: event.payload.code });
  });

  return () => {
    offMessage();
    offExit();
  };
}

export async function subscribeToFileDrops(
  onPath: (path: string) => void,
): Promise<() => void> {
  if (!isTauriRuntime()) {
    return () => {};
  }
  const offDrop = await getCurrentWebviewWindow().onDragDropEvent((event) => {
    if (event.payload.type === "drop" && event.payload.paths.length > 0) {
      onPath(event.payload.paths[0] ?? "");
    }
  });

  return () => {
    offDrop();
  };
}

function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}
