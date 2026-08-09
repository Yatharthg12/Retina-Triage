const RT = {
  async fetchJSON(url, options = {}, timeout = 20000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(url, {...options, signal: controller.signal});
      const payload = await response.json();
      if (!response.ok) {
        const error = new Error(payload.error?.message || `Request failed (${response.status})`);
        error.payload = payload; error.status = response.status; throw error;
      }
      return payload.data;
    } finally { clearTimeout(timer); }
  },
  toast(message, kind = "") {
    const item = document.createElement("div"); item.className = `toast ${kind}`; item.textContent = message;
    document.querySelector("#toasts").append(item); setTimeout(() => item.remove(), 4200);
  },
  badge(priority) {
    const cls = priority?.startsWith("URGENT") ? "urgent" : priority?.includes("HIGH") ? "warning" : priority?.includes("REVIEW") ? "review" : "routine";
    const span = document.createElement("span"); span.className = `badge ${cls}`; span.textContent = priority || "—"; return span;
  },
  percent(value) { return value == null ? "—" : `${(Number(value) * 100).toFixed(1)}%`; },
  el(tag, text, className) { const node = document.createElement(tag); if (text != null) node.textContent = text; if (className) node.className = className; return node; }
};
document.querySelector("#menuButton")?.addEventListener("click", () => document.querySelector(".sidebar").classList.toggle("open"));
const clock = document.querySelector("#clock");
if (clock) { const tick = () => clock.textContent = new Date().toLocaleString([], {dateStyle:"medium", timeStyle:"short"}); tick(); setInterval(tick, 60000); }

