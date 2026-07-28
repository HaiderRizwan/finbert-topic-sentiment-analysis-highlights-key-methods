document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-upload");
    const preview = document.getElementById("file-preview");
    const btn = document.getElementById("analyze-btn");
    const modelSelect = document.getElementById("model-select");

    const resultsPanel = document.getElementById("results-panel");
    const primaryStatusCard = document.getElementById("primary-status");
    const primaryClassObj = document.getElementById("primary-class");
    const primaryConfObj = document.getElementById("primary-confidence");

    let currentFile = null;

    // --- Drag and Drop Logic --- //
    dropZone.addEventListener("click", () => fileInput.click());
    
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });
    
    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });
    
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.match('image.*')) {
            alert("Please select a valid image file.");
            return;
        }
        currentFile = file;
        preview.textContent = `Attached: ${file.name}`;
        preview.classList.remove("hidden");
        btn.removeAttribute("disabled");
        resultsPanel.classList.add("hidden"); // hide old results
    }

    // --- API Request Logic --- //
    btn.addEventListener("click", async () => {
        if (!currentFile) return;

        // UI State: Loading
        btn.setAttribute("disabled", "true");
        btn.querySelector(".btn-text").textContent = "Analyzing Geometry...";
        btn.querySelector(".btn-spinner").classList.remove("hidden");
        resultsPanel.classList.add("hidden");

        const formData = new FormData();
        formData.append("file", currentFile);
        formData.append("model", modelSelect.value);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });
            const data = await response.json();
            
            if (data.success) {
                renderResults(data);
            } else {
                alert(`Analysis Failed: ${data.error}`);
            }
        } catch (error) {
            alert(`Connection Error: ${error.message}`);
        } finally {
            // UI State: Reset
            btn.removeAttribute("disabled");
            btn.querySelector(".btn-text").textContent = "Initialize Run";
            btn.querySelector(".btn-spinner").classList.add("hidden");
        }
    });

    // --- Metrics Rendering Logic --- //
    function renderResults(data) {
        const pClass = data.prediction; // 'Safe', 'Violence', 'Erotism'
        
        // Setup Primary Card
        primaryClassObj.textContent = pClass;
        primaryConfObj.textContent = `${data.confidence}% Confidence`;
        
        // Remove old dynamic color classes
        primaryStatusCard.className = "status-card"; 
        primaryStatusCard.classList.add(`status-${pClass.toLowerCase()}`);

        // Update Progress Bars dynamically
        updateBar("safe", data.details["Safe"]);
        updateBar("violence", data.details["Violence"]);
        updateBar("erotism", data.details["Erotism"]);

        // Debug Metrics
        document.getElementById("latency-ms").textContent = data.latency_ms || "N/A";
        document.getElementById("model-loaded").textContent = data.model_used.toUpperCase();

        // Uncertainty Alert
        const alertBox = document.getElementById("uncertainty-alert");
        if (data.confidence < 65.0) {
            alertBox.classList.remove("hidden");
        } else {
            alertBox.classList.add("hidden");
        }

        // Reveal panel using CSS Animation
        resultsPanel.classList.remove("hidden");
    }

    function updateBar(id_suffix, percentage) {
        const bar = document.getElementById(`bar-${id_suffix}`);
        const text = document.getElementById(`txt-${id_suffix}`);
        
        // Reset to 0 briefly so the transition retriggers
        bar.style.width = "0%";
        
        // setTimeout forces exactly enough DOM reflow to trigger the smooth CSS width transform
        setTimeout(() => {
            bar.style.width = `${percentage}%`;
            text.textContent = `${percentage}%`;
        }, 50);
    }
});
