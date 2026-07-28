from pathlib import Path

html_content = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SafeToon UML</title>

<!-- Mermaid CDN -->
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
mermaid.initialize({
    startOnLoad: true,
    flowchart: { useMaxWidth: true, htmlLabels: true }
});
</script>

<style>
/* ===== A4 PAGE SETUP ===== */
@page {
    size: A4;
    margin: 10mm;
}

body {
    margin: 0;
    padding: 10mm;
    background: white;
    font-family: Arial;
}

/* A4 container */
.a4 {
    width: 210mm;
    height: 297mm;
    margin: auto;
    padding: 10mm;
    box-sizing: border-box;
    border: 1px solid #ddd;
}

/* Make diagram fit */
.mermaid {
    transform: scale(0.85);
    transform-origin: top left;
}
</style>

</head>

<body>
<div class="a4">

<div class="mermaid">
graph TB

SafeToon["SafeToon System"]

FYP1["FYP-1 (Completed)"]
FYP2["FYP-2 (Planned)"]

SafeToon --> FYP1
SafeToon --> FYP2

subgraph F1["FYP-1 Modules"]
direction TB

Ingestion["Video Ingestion"]
Dataset["Dataset Mgmt"]
Analyzer["Frame Analyzer"]
Training["Training & Eval"]

FYP1 --> Ingestion
FYP1 --> Dataset
FYP1 --> Analyzer
FYP1 --> Training

Ingestion --> Val["Validation"]
Ingestion --> Ext["Frame/Audio Extraction"]
Ingestion --> Meta["Metadata Gen"]

Dataset --> Tax["3-Class Taxonomy"]
Dataset --> Ann["Annotation Guides"]

Analyzer --> Pre["Preprocessing"]
Analyzer --> Inf["Inference Endpoint"]

Training --> Models["NASNet / DINOv2"]
Training --> Metrics["Metrics & Curves"]

end

subgraph F2["FYP-2 Modules"]
direction TB

Audio["Audio Analyzer"]
Cut["Surgical Cut"]
UI["Dashboard"]

FYP2 --> Audio
FYP2 --> Cut
FYP2 --> UI

Audio --> Prof["Profanity Detect"]
Audio --> Em["Emotion Detect"]

Cut --> Blur["Scene Blurring"]
Cut --> Mute["Audio Muting"]
Cut --> Recon["Video Reconstruction"]

end

</div>

</div>
</body>
</html>
"""

# Save file
file_path = Path("safetoon_a4_diagram.html")
file_path.write_text(html_content, encoding="utf-8")

print("HTML file created:", file_path.resolve())