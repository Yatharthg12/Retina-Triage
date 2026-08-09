const batchInput=document.querySelector("#batchInput"),batchButton=document.querySelector("#batchButton");
let currentBatch=[];
batchInput.addEventListener("change",()=>{const n=batchInput.files.length;document.querySelector("#batchSelected").textContent=n;batchButton.disabled=!n||n>20;if(n>20)RT.toast("A batch may contain at most 20 images.","error");});
batchButton.addEventListener("click",async()=>{
  batchButton.disabled=true; const form=new FormData(); [...batchInput.files].forEach(f=>form.append("images",f));
  try{const data=await RT.fetchJSON("/api/predict/batch",{method:"POST",body:form},120000);renderBatch(data.results);}
  catch(error){if(error.payload?.data?.results)renderBatch(error.payload.data.results);RT.toast(error.message,"error");}
  finally{batchButton.disabled=false;}
});
function renderBatch(results){
  const rank={"URGENT – HIGH PRIORITY":0,"HIGH PRIORITY":1,"RETAKE / MANUAL REVIEW":2,"SPECIALIST REVIEW":3,"FOLLOW-UP":4,"ROUTINE":5};
  results=[...results].sort((a,b)=>(rank[a.data?.triage?.priority]??9)-(rank[b.data?.triage?.priority]??9));
  currentBatch=results;document.querySelector("#batchExport").disabled=!results.length;
  const body=document.querySelector("#batchRows");body.textContent="";let complete=0,review=0,errors=0;
  results.forEach(item=>{const row=body.insertRow();row.dataset.name=item.filename.toLowerCase();row.insertCell().textContent=item.filename;
    row.insertCell().append(RT.el("span",item.success?"Completed":item.error||"Failed",`badge ${item.success?"ready":"urgent"}`));
    if(item.data){row.insertCell().textContent=`${Math.round(item.data.quality.quality_score*100)} / 100`;row.insertCell().textContent=item.data.prediction?`Grade ${item.data.prediction.grade}`:"Withheld";row.insertCell().append(RT.badge(item.data.triage.priority));if(item.data.triage.manual_review)review++;if(item.success)complete++;else errors++;}
    else{row.insertCell().textContent="—";row.insertCell().textContent="—";row.insertCell().textContent="—";errors++;}
  });document.querySelector("#batchCompleted").textContent=complete;document.querySelector("#batchReview").textContent=review;document.querySelector("#batchErrors").textContent=errors;
}
document.querySelector("#batchSearch").addEventListener("input",e=>{const q=e.target.value.toLowerCase();document.querySelectorAll("#batchRows tr").forEach(r=>r.hidden=!(r.dataset.name||"").includes(q));});
document.querySelector("#batchExport").addEventListener("click",()=>{
  const quote=value=>`"${String(value??"").replaceAll('"','""')}"`;
  const rows=[["filename","status","quality_score","grade","priority","manual_review"],
    ...currentBatch.map(item=>[item.filename,item.success?"completed":item.error,item.data?.quality?.quality_score,item.data?.prediction?.grade,item.data?.triage?.priority,item.data?.triage?.manual_review])];
  const csv=rows.map(row=>row.map(quote).join(",")).join("\r\n");
  const link=document.createElement("a");link.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));link.download="retina-triage-batch.csv";link.click();URL.revokeObjectURL(link.href);
});
