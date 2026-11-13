# Overview

Offsider is an end-to-end computer vision and rule-evaluation system designed to eliminate parking uncertainty.

Delivery drivers operate under tight window drops, where reading complex parking signs from the cab is confusing and error-prone, often resorting to one's own intuition to validate parking. 

Parking signage changes frequently, especially in dense areas. A high-pressure environment, combined with lack of present knowledge around local parking laws leads to an excess of parking tickets for drivers.

Offsider pairs object detection with optical character recognition (OCR) to frame parking signs, transcribe regulatory text, and validate parking permissions against the current vehicle classification and timestamp, in real-time.

---

## Architecture

```mermaid
graph TD
    subgraph In-Vehicle Edge System
        A["Live Dashcam Video Stream"] --> B["Vehicle State Monitor<br>(Handbrake / Shift to Park)"]
        B -->|Trigger Event| C["Frame Capture Buffer"]
        C --> D["YOLOv8n Object Detector<br>(Sign Localization)"]
        D --> E["EasyOCR Engine<br>(Text Extraction)"]
        E --> F["Deterministic Rule Parser<br>(Time, Day & Limit Match)"]
        F --> G["Driver Dashboard UI Feedback"]
    end

    style A fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1
    style B fill:#FFF9C4,stroke:#FBC02D,stroke-width:2px,color:#F57F17
    style C fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20
    style D fill:#EDE7F6,stroke:#512DA8,stroke-width:2px,color:#311B92
    style E fill:#EDE7F6,stroke:#512DA8,stroke-width:2px,color:#311B92
    style F fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20
    style G fill:#E1F5FE,stroke:#0288D1,stroke-width:2px,color:#01579B
```


## Demo User Flow
```mermaid
graph TD
    Start([System Idle / Monitoring]) --> Trigger[Simulated Event Trigger:\nSelect Scenario & Engage Handbrake]
    Trigger --> RunPipeline[Execute YOLOv8n + EasyOCR Pipeline]
    RunPipeline --> Check1{Is there a frame with a clear parking sign?}
    Check1 -->|No| Manual[Display 'CHECK MANUALLY' on Dashboard]
    Check1 -->|Yes| Parse[Evaluate via Deterministic Rule Engine]
    Parse --> Check2{Is Parking Allowed?}
    Check2 -->|Yes| Clear[Display 'CLEAR / LEGAL' on Dashboard]
    Check2 -->|No| Restrict[Display 'RESTRICTED / FINE RISK' on Dashboard]
    Clear --> End([Standby for Next Event])
    Restrict --> End
    Manual --> End

    style Start fill:#ECEFF1,stroke:#455A64,stroke-width:2px,color:#263238
    style Trigger fill:#FFF9C4,stroke:#FBC02D,stroke-width:2px,color:#F57F17
    style RunPipeline fill:#EDE7F6,stroke:#512DA8,stroke-width:2px,color:#311B92
    style Parse fill:#E1F5FE,stroke:#0288D1,stroke-width:2px,color:#01579B
    style Check2 fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px,color:#E65100
    style Clear fill:#C8E6C9,stroke:#388E3C,stroke-width:2px,color:#1B5E20
    style Restrict fill:#FFCDD2,stroke:#D32F2F,stroke-width:2px,color:#B71C1C
    style Manual fill:#ffe354,stroke:#aa9f67,stroke-width:2px,color:#988a42
    style End fill:#ECEFF1,stroke:#455A64,stroke-width:2px,color:#263238
```


---

## Repository Structure

```text
offsider/
├── data/                  # Local training sets (ignored by git)
├── docs/
|   ├── assets/            # Images
│   └── index.html         # Project landing page
├── fixtures/              # Test fixtures and unit test inputs
│   └── sample_crops/      # Pre-cropped signage for mock evals
├── models/                # Local model (ignored by git)
│   └── best.pt
├── notebooks/             # Analysis, training, evals
│   └── training.ipynb
├── src/                   # Core application logic
│   ├── pipeline.py        # Sign detector
|   ├── ocr.py             # Text extractor
│   └── rules.py           # Rule parser
├── streamlit_app/         # Streamlit prototype
|   ├── app.py
|   └── requirements.txt
├── app.py
├── pyproject.toml         # Environment configuration
└── README.md
```

# Getting Started

### Prerequisites
  - Python 3.12 or higher
  - Git

### Installation

1. Clone the repo:
   
    `git clone https://github.com/offsider-labs/offsider.git`

2. Create and activate virtual environment:

    `uv venv && uv sync`

### Model Weights Setup

Model weight binaries (.pt) are tracked outside of Git.

1. Download the latest trained checkpoint from our [latest pre-release](https://github.com/offsider-labs/offsider/releases/).
2. Place the weights file inside `models/` directory:

```bash
mkdir -p models
mv ~/Downloads/best.pt models/best.pt
```

## Usage

### Running the Interactive UI

`streamlit run steamlit_app/app.py`

## Contributors & Team

__David Van Kriedt__ - Main Developer

__Aryan Gupta__ - Main Developer
