# TruthGuard_AI Simplified Sequence Diagram

This is a simplified, high-level version of the TruthGuard_AI runtime flow.
It groups the detailed detectors into a few major runtime components so the
overall behavior is easier to present in reports and documentation.

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant Core as TruthGuard Core Services
    participant Worker as Isolated Workers
    participant Store as Local Models & Storage
    participant Ext as External APIs / Hugging Face

    User->>UI: Open app and choose detector
    UI->>Core: Validate input and route request
    Core->>Store: Load cached settings and local models
    opt Heavy transformer or deepfake analysis
        Core->>Worker: Start isolated analysis
        Worker->>Store: Read temp files and heavy model assets
        opt Online verification or remote model needed
            Worker->>Ext: Fetch API data or HF model files
        end
        Worker-->>Core: Return safe prediction payload
    end
    Core-->>UI: Build verdict, confidence, and charts
    UI-->>User: Display result and evidence
```

Covered modules:
- Deepfake detection
- Fake news detection
- Sentiment analysis
- Toxicity detection
