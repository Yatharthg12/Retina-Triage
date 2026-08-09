(async () => {
  try {
    const data = await RT.fetchJSON("/api/dashboard/summary");
    document.querySelector("#totalScreenings").textContent = data.total_screenings;
    document.querySelector("#highPriority").textContent = data.high_priority;
    document.querySelector("#manualReview").textContent = data.manual_review;
    document.querySelector("#poorQuality").textContent = data.poor_quality;
    document.querySelector("#avgLatency").textContent = `${data.average_processing_time_ms} ms`;
    document.querySelector("#modelVersion").textContent = data.model.version;
    const badge = document.querySelector("#modelBadge"); badge.textContent = data.model.status; badge.className = `badge ${data.model.available ? "ready" : "urgent"}`;
    const body = document.querySelector("#recentQueue"); body.textContent = "";
    if (!data.recent.length) { const row = body.insertRow(); const cell = row.insertCell(); cell.colSpan = 5; cell.append(RT.el("div", "No screenings recorded yet.", "empty-inline")); }
    data.recent.forEach(item => {
      const row = body.insertRow();
      row.insertCell().textContent = item.case_id || item.screening_id.slice(0, 8);
      row.insertCell().textContent = item.predicted_grade == null ? "No result" : `Grade ${item.predicted_grade}`;
      row.insertCell().append(RT.badge(item.priority));
      row.insertCell().textContent = RT.percent(item.confidence);
      row.insertCell().textContent = new Date(item.created_at).toLocaleDateString();
    });
    const bars = document.querySelector("#severityBars"); bars.textContent = "";
    const max = Math.max(1, ...Object.values(data.severity_distribution));
    for (let grade=0; grade<5; grade++) {
      const count = data.severity_distribution[String(grade)] || 0;
      const row = RT.el("div", null, "severity-bar"); row.append(RT.el("b", `G${grade}`));
      const track = RT.el("i"); const fill = RT.el("span"); fill.style.width = `${count/max*100}%`; track.append(fill);
      row.append(track, RT.el("em", String(count))); bars.append(row);
    }
  } catch (error) { RT.toast(error.message, "error"); }
})();

