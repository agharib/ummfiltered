import "@testing-library/jest-dom/vitest";

const storage = new Map<string, string>();

const memoryStorage: Storage = {
  clear() {
    storage.clear();
  },
  getItem(key) {
    return storage.has(key) ? storage.get(key) ?? null : null;
  },
  key(index) {
    return Array.from(storage.keys())[index] ?? null;
  },
  get length() {
    return storage.size;
  },
  removeItem(key) {
    storage.delete(key);
  },
  setItem(key, value) {
    storage.set(key, String(value));
  },
};

if (typeof window !== "undefined") {
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: memoryStorage,
  });
}
