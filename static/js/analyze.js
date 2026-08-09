const input = document.querySelector("#imageInput"), dropzone = document.querySelector("#dropzone");
const button = document.querySelector("#analyzeButton"), preview = document.querySelector("#preview");
function selectFile(file) {
  if (!file) return;
  if (!["image/jpeg","image/png"].includes(file.type) || file.size > 12*1024*1024) { RT.toast("Choose a JPG or PNG smaller than 12 MB.", "error"); return; }
  const transfer = new DataTransfer(); transfer.items.add(file); input.files = transfer.files;
  preview.src = URL.createObjectURL(file); preview.hidden = false;
  [...dropzone.children].forEach(node => { if (node !== input && node !== preview) node.hidden = true; });
  button.disabled = false;
}
async function loadDemoSamples() {
  const container = document.querySelector("#demoSampleButtons");
  try {
    const manifest = await RT.fetchJSON("/api/demo/samples");
    container.textContent = "";
    document.querySelector("#demoSampleNotice").textContent = manifest.notice;
    manifest.samples.forEach(item => {
      const sampleButton = RT.el("button", item.title, "demo-sample-button");
      sampleButton.type = "button";
      sampleButton.title = `${item.source_description} ${item.presentation_use}`;
      sampleButton.addEventListener("click", async () => {
        sampleButton.disabled = true;
        try {
          const response = await fetch(item.url);
          if (!response.ok) throw new Error("The sample image could not be loaded.");
          const blob = await response.blob();
          const file = new File([blob], item.filename, {type: blob.type || "image/jpeg"});
          selectFile(file);
          document.querySelector("#caseId").value = item.case_id;
          RT.toast(`${item.title} selected.`);
        } catch (error) {
          RT.toast(error.message, "error");
        } finally {
          sampleButton.disabled = false;
        }
      });
      container.append(sampleButton);
    });
    if (!manifest.samples.length) container.append(RT.el("small", "Sample pack is not installed."));
  } catch (error) {
    container.textContent = "";
    container.append(RT.el("small", "Sample pack unavailable."));
  }
}
input.addEventListener("change", () => selectFile(input.files[0]));
["dragenter","dragover"].forEach(event => dropzone.addEventListener(event, e => {e.preventDefault(); dropzone.classList.add("drag");}));
["dragleave","drop"].forEach(event => dropzone.addEventListener(event, e => {e.preventDefault(); dropzone.classList.remove("drag");}));
dropzone.addEventListener("drop", e => selectFile(e.dataTransfer.files[0]));
function renderResult(result, modelError) {
  const area = document.querySelector("#resultArea"); area.textContent = "";
  const quality = result.quality, prediction = result.prediction;
  const hero = RT.el("article", null, "result-hero");
  const left = RT.el("div"); left.append(RT.el("small", quality.gradable ? "QUALITY GATE PASSED" : "QUALITY GATE FAILED"));
  left.append(RT.el("h2", prediction ? `Grade ${prediction.grade} · ${prediction.label}` : "No disease prediction"));
  const outputLabel = prediction
    ? (prediction.confidence_kind === "temperature_calibrated" ? "Calibrated research screening output" : "Uncalibrated research screening output")
    : quality.decision_reason;
  left.append(RT.el("small", modelError || outputLabel));
  hero.append(left, RT.badge(result.triage.priority)); area.append(hero);
  const columns = RT.el("section", null, "quality-columns");
  const q = RT.el("article", null, "panel"); q.append(RT.el("p", "IMAGE QUALITY", "eyebrow"), RT.el("h2", "Transparent quality gate"));
  q.append(RT.el("div", `${Math.round(quality.quality_score*100)} / 100`, "quality-score"));
  q.append(RT.el("p", `Acceptance threshold: ${Math.round(quality.minimum_score*100)} / 100`, "quality-threshold"));
  q.append(RT.el("p", quality.gradable ? "Composite threshold passed." : "Automated grading is withheld.", ""));
  const observations = quality.observations || quality.issues || [];
  if (observations.length) { q.append(RT.el("small", "Heuristic observations", "observation-label")); const ul = RT.el("ul", null, "issues"); observations.forEach(x => ul.append(RT.el("li", x))); q.append(ul); }
  q.append(RT.el("p", quality.disclaimer, "evidence-warning"));
  const triageHeading = !quality.gradable ? "Manual approval required" : (result.triage.manual_review ? "Manual review required" : "Automated routing");
  const p = RT.el("article", null, "panel"); p.append(RT.el("p", "TRIAGE", "eyebrow"), RT.el("h2", triageHeading));
  const reasons = RT.el("ul", null, "issues"); result.triage.reasons.forEach(x => reasons.append(RT.el("li", x))); p.append(reasons);
  columns.append(q,p); area.append(columns);
  if (prediction) {
    const confidenceLabel = prediction.confidence_kind === "temperature_calibrated" ? "CALIBRATED OUTPUT" : "UNCALIBRATED MODEL OUTPUT";
    const probs = RT.el("article", null, "panel"); probs.style.marginTop = "14px"; probs.append(RT.el("p",confidenceLabel,"eyebrow"),RT.el("h2",`Confidence ${RT.percent(prediction.confidence)} · Referable ${RT.percent(prediction.referable_probability)} · High-risk ${RT.percent(prediction.high_risk_probability)}`));
    if (prediction.confidence_kind !== "temperature_calibrated") probs.append(RT.el("p","Confidence is raw softmax output and is not clinically calibrated.","evidence-note"));
    const list = RT.el("div",null,"probability-list"); list.style.marginTop="16px";
    prediction.probabilities.forEach((value, grade) => { const row=RT.el("div",null,"prob-row"); row.append(RT.el("b",`G${grade}`)); const track=RT.el("i"); const fill=RT.el("span"); fill.style.width=`${value*100}%`; track.append(fill); row.append(track,RT.el("em",RT.percent(value))); list.append(row); }); probs.append(list); area.append(probs);
  }
  if (result.explanation?.overlay) {
    const explain = RT.el("article", null, "panel"); explain.style.marginTop = "14px";
    explain.append(RT.el("p", "VISUAL EXPLANATION", "eyebrow"), RT.el("h2", "Grad-CAM influence map"));
    const image = RT.el("img"); image.src = result.explanation.overlay; image.alt = "Grad-CAM influence overlay on processed retinal image";
    image.style.cssText = "display:block;max-width:420px;width:100%;border-radius:10px;margin:14px auto";
    const label = RT.el("label"); label.textContent = "Overlay intensity ";
    const slider = document.createElement("input"); slider.type = "range"; slider.min = "20"; slider.max = "100"; slider.value = "100";
    slider.setAttribute("aria-label", "Grad-CAM overlay intensity"); slider.addEventListener("input", () => image.style.opacity = slider.value / 100);
    label.append(slider); explain.append(image, label, RT.el("p", result.explanation.disclaimer)); area.append(explain);
  }
}
button.addEventListener("click", async () => {
  button.disabled = true; const progress = document.querySelector("#progressTrack"); progress.hidden = false;
  const data = new FormData(); data.append("image", input.files[0]); data.append("case_id", document.querySelector("#caseId").value); data.append("include_gradcam", "true");
  try { renderResult(await RT.fetchJSON("/api/predict", {method:"POST",body:data}, 60000)); RT.toast("Screening completed."); }
  catch (error) { if (error.payload?.data) renderResult(error.payload.data, error.message); else RT.toast(error.message,"error"); }
  finally { progress.hidden=true; button.disabled=false; }
});
loadDemoSamples();
