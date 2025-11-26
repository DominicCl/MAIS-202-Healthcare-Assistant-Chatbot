(function () {
  const textarea = document.getElementById("symptomInput");
  const sendBtn = document.getElementById("sendBtn");
  const resetBtn = document.getElementById("resetBtn");
  const modelOutput = document.getElementById("modelOutput");
  const status = document.getElementById("status");
  const apiIndicator = document.getElementById("apiIndicator");

  //  FIXED: Force backend URL
  const apiBase = "http://127.0.0.1:5000";
  apiIndicator.textContent = `API: ${apiBase}`;

  const route = (path) => `${apiBase}${path}`;



  function setStatus(message, type = "") {
    status.textContent = message || "";
    status.className = `status ${type}`.trim();
  }

  async function sendMessage() {
    const text = textarea.value.trim();
    if (!text) {
      setStatus("Please describe your symptoms before sending.", "error");
      return;
    }

    sendBtn.disabled = true;
    resetBtn.disabled = true;
    setStatus("Sending symptoms to chatbot...");

    try {
      const response = await fetch(route("/api/chat"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || "Chatbot returned an error");
      }

      const data = await response.json();
      renderOutput(data);

      const reply =
 	 "Detected symptoms:\n" +
  	(data.symptoms.join(", ") || "None") +
  	"\n\nPredictions:\n" +
  	data.predictions
    	.map(p => `${p.disease} — ${(p.probability * 100).toFixed(1)}%`)
    	.join("\n");

      function renderOutput(data) {
  const output = document.getElementById("modelOutput");
  output.innerHTML = "";

  // Symptoms section
  if (data.symptoms && data.symptoms.length > 0) {
    const symBox = document.createElement("div");
    symBox.innerHTML = `<h3>Detected Symptoms</h3>`;
    const symList = document.createElement("div");
    symList.className = "tag-list";

    data.symptoms.forEach(sym => {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = sym.replace(/_/g, " ");
      symList.appendChild(tag);
    });

    symBox.appendChild(symList);
    output.appendChild(symBox);
  }

  // Predictions section
  if (data.predictions && data.predictions.length > 0) {
    const predBox = document.createElement("div");
    predBox.innerHTML = `<h3>Predictions</h3>`;

    data.predictions.forEach(pred => {
      const row = document.createElement("div");
      row.className = "prediction-row";

      const probPercent = Math.round(pred.probability * 100);

      row.innerHTML = `
        <div class="pred-label">${pred.disease.replace(/_/g, " ")}</div>
        <div class="pred-bar">
          <div class="pred-bar-fill" style="width:${probPercent}%;"></div>
        </div>
        <div class="pred-value">${probPercent}%</div>
      `;

      predBox.appendChild(row);
    });

    output.appendChild(predBox);
  }
}

      textarea.value = "";
      setStatus("Response received.", "ok");
    } catch (err) {
      console.error(err);
      setStatus(err.message || "Could not reach the chatbot.", "error");
    } finally {
      sendBtn.disabled = false;
      resetBtn.disabled = false;
    }
  }

  async function resetConversation() {
    sendBtn.disabled = true;
    resetBtn.disabled = true;
    setStatus("Resetting chatbot...");

    try {
      const response = await fetch(route("/api/reset"), { method: "POST" });
      if (!response.ok) {
        throw new Error("Reset failed");
      }
      modelOutput.textContent = "The model response will appear here.";
      textarea.value = "";
      setStatus("Conversation reset.", "ok");
    } catch (err) {
      console.error(err);
      setStatus(err.message || "Unable to reset conversation.", "error");
    } finally {
      sendBtn.disabled = false;
      resetBtn.disabled = false;
    }
  }

  sendBtn.addEventListener("click", sendMessage);
  resetBtn.addEventListener("click", resetConversation);
})();
