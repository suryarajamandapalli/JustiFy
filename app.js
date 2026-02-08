document.getElementById("analyzeBtn").addEventListener("click", () => {
  document.getElementById("mlOutput").classList.remove("hidden");

  // Temporary placeholders (will be replaced by ML output)
  document.getElementById("caseType").innerText = "—";
  document.getElementById("confidence").innerText = "—";
  document.getElementById("keySignals").innerText = "—";
  document.getElementById("legalDirection").innerText = "—";
});