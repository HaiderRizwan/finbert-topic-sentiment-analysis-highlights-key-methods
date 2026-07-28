document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone-video");
    const fileUpload = document.getElementById("video-upload");
    const filePreview = document.getElementById("video-preview");
    const ingestBtn = document.getElementById("ingest-btn");
    const btnText = ingestBtn.querySelector(".btn-text");
    const btnSpinner = ingestBtn.querySelector(".btn-spinner");
    const resultsPanel = document.getElementById("ingest-results-panel");
    const fpsInput = document.getElementById("fps-input");
    const outputDirInput = document.getElementById("output-dir");

    const elStatus = document.getElementById("ingest-status");
    const elJobId = document.getElementById("job-id-text");
    const elDuration = document.getElementById("txt-duration");
    const elResolution = document.getElementById("txt-resolution");
    const elFrames = document.getElementById("txt-frames");
    const elAudio = document.getElementById("txt-audio");
    const elOutVideo = document.getElementById("out-video");
    const elOutFrames = document.getElementById("out-frames");
    const elOutAudio = document.getElementById("out-audio");
    const primaryStatusCard = document.getElementById("primary-status");

    let selectedFiles = [];

    // --- Drag and Drop ---
    dropZone.addEventListener("click", () => fileUpload.click());

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
        if (e.dataTransfer.files.length) {
            handleFileSelection(e.dataTransfer.files);
        }
    });

    fileUpload.addEventListener("change", (e) => {
        if (e.target.files.length) {
            handleFileSelection(e.target.files);
        }
    });

    function handleFileSelection(files) {
        selectedFiles = Array.from(files).filter(f => f.type.startsWith("video/"));
        if (selectedFiles.length === 0) {
            alert("Please upload valid video files.");
            return;
        }
        
        if (selectedFiles.length === 1) {
            filePreview.textContent = `Selected: ${selectedFiles[0].name} (${(selectedFiles[0].size / 1024 / 1024).toFixed(2)} MB)`;
        } else {
            const totalSize = selectedFiles.reduce((acc, f) => acc + f.size, 0);
            filePreview.textContent = `Selected ${selectedFiles.length} videos (${(totalSize / 1024 / 1024).toFixed(2)} MB) - Ready to process`;
        }
        
        filePreview.classList.remove("hidden");
        ingestBtn.disabled = false;
        resultsPanel.classList.add("hidden");
        document.getElementById("progress-container").classList.add("hidden");
    }

    // --- Ingestion Action ---
    ingestBtn.addEventListener("click", async () => {
        if (selectedFiles.length === 0) return;

        // UI Loading State
        ingestBtn.disabled = true;
        btnText.textContent = "Processing Videos...";
        btnSpinner.classList.remove("hidden");
        resultsPanel.classList.add("hidden");
        
        const progContainer = document.getElementById("progress-container");
        const progText = document.getElementById("progress-text");
        const progPerc = document.getElementById("progress-percentage");
        const progFill = document.getElementById("progress-bar-fill");
        progContainer.classList.remove("hidden");

        let cumulativeFrames = 0;
        let cumulativeAudio = 0;
        let cumulativeDuration = 0.0;
        let successCount = 0;
        let failCount = 0;
        
        for (let i = 0; i < selectedFiles.length; i++) {
            const file = selectedFiles[i];
            
            // Update progress pre-processing
            const pct = Math.round((i / selectedFiles.length) * 100);
            progText.textContent = `Processing video ${i + 1} of ${selectedFiles.length} (${file.name})...`;
            progPerc.textContent = `${pct}%`;
            progFill.style.width = `${pct}%`;

            const formData = new FormData();
            formData.append("file", file);
            formData.append("fps", fpsInput.value);
            formData.append("output_dir", outputDirInput.value);

            try {
                const response = await fetch("/api/ingest", {
                    method: "POST",
                    body: formData
                });
                const data = await response.json();

                if (data.success) {
                    successCount++;
                    cumulativeFrames += data.n_frames || 0;
                    cumulativeAudio += data.n_audio_chunks || 0;
                    if (data.video_metadata && data.video_metadata.duration) {
                        cumulativeDuration += data.video_metadata.duration;
                    }
                } else {
                    failCount++;
                    console.error("Ingestion failed for", file.name, data.error);
                }
            } catch (error) {
                failCount++;
                console.error("Fetch error for", file.name, error);
            }
            
            // Update progress post-processing
            const pctDone = Math.round(((i + 1) / selectedFiles.length) * 100);
            progPerc.textContent = `${pctDone}%`;
            progFill.style.width = `${pctDone}%`;
        }

        // Revert UI state
        ingestBtn.disabled = false;
        btnText.textContent = "Run Ingestion";
        btnSpinner.classList.add("hidden");
        progText.textContent = "Processing complete.";

        showResults(successCount, failCount, cumulativeFrames, cumulativeAudio, cumulativeDuration, selectedFiles.length);
    });

    function showResults(successes, fails, frames, audioChunks, duration, total) {
        if (successes > 0) {
            primaryStatusCard.style.borderColor = fails > 0 ? "rgba(234, 179, 8, 0.5)" : "rgba(74, 222, 128, 0.5)"; // yellow if mixed, green if perfect
            elStatus.textContent = fails > 0 ? "PARTIAL SUCCESS" : "SUCCESS";
            elStatus.style.color = fails > 0 ? "#eab308" : "var(--accent-safe)";
            elStatus.style.textShadow = fails > 0 ? "0 0 15px rgba(234, 179, 8, 0.5)" : "0 0 15px rgba(74, 222, 128, 0.5)";
        } else {
            primaryStatusCard.style.borderColor = "rgba(239, 68, 68, 0.5)";
            elStatus.textContent = "ERROR";
            elStatus.style.color = "var(--accent-violence)";
            elStatus.style.textShadow = "0 0 15px rgba(239, 68, 68, 0.5)";
        }

        elJobId.textContent = `Processed ${successes} out of ${total} videos. (Failed: ${fails})`;
        elDuration.textContent = `${duration.toFixed(2)}s (Total)`;
        elResolution.textContent = `Multiple`; // Irrelevant when combining multiple
        elFrames.textContent = frames;
        elAudio.textContent = audioChunks;

        elOutVideo.textContent = "Multiple files";
        elOutFrames.textContent = `${outputDirInput.value || 'outputs/'}[VideoName]/frames/`;
        elOutAudio.textContent = `${outputDirInput.value || 'outputs/'}[VideoName]/audio/`;

        resultsPanel.classList.remove("hidden");
    }
});
