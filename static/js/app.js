document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".needs-validation").forEach((form) => {
    form.addEventListener("submit", (event) => {
      if (!form.checkValidity()) {
        event.preventDefault();
        event.stopPropagation();
      }
      form.classList.add("was-validated");
    });
  });

  const predictionForm = document.getElementById("predictionForm");
  if (predictionForm) {
    predictionForm.addEventListener("submit", (event) => {
      if (!predictionForm.checkValidity()) {
        event.preventDefault();
        event.stopPropagation();
      }
      predictionForm.classList.add("was-validated");
    });
  }
});
