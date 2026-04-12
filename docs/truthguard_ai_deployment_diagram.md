# TruthGuard_AI Deployment Diagram

This deployment view shows how the current project is typically arranged at runtime.
It focuses on deployed nodes, process boundaries, local storage, and optional external
services.

```mermaid
flowchart LR
    subgraph client["Client Device"]
        user["User / Analyst / Admin"]
        browser["Browser"]
        user --> browser
    end

    subgraph host["TruthGuard AI Host"]
        subgraph web["Web Application Process"]
            st["Streamlit App<br/>app.py"]
            auth["Auth + Session State"]
            cache["Lazy Cache / Component Loader"]
            ocr["OCR / Input Processing"]
        end

        subgraph services["Core Analysis Services"]
            fn["Fake News Detector"]
            rv["Realtime News Verifier"]
            sa["Sentiment Analyzer"]
            aa["Aspect Sentiment Analyzer"]
            tx["Toxicity Detector"]
        end

        subgraph workers["Isolated Worker Processes"]
            dw["Deepfake Worker<br/>scripts/isolated_deepfake_worker.py"]
            tw["Transformer Worker<br/>scripts/isolated_transformer_worker.py"]
        end

        subgraph storage["Local Storage on Host"]
            env[".env<br/>admin + API secrets"]
            userdb["users.json<br/>local user accounts"]
            models["models/<br/>.h5 / .weights.h5 / .pkl / metadata"]
            temp["temp/streamlit_artifacts<br/>uploads + generated artifacts"]
            data["datasets/<br/>training / offline data"]
        end
    end

    subgraph external["Optional External Services"]
        hf["Hugging Face Hub"]
        news["NewsAPI + Google News RSS + article pages"]
        translate["Google Translate via googletrans"]
    end

    browser <-->|"HTTP / HTTPS"| st

    st --> auth
    st --> cache
    st --> ocr
    st --> fn
    st --> rv
    st --> sa
    st --> aa
    st --> tx
    st --> dw

    auth --> env
    auth --> userdb

    cache --> fn
    cache --> rv
    cache --> sa
    cache --> aa
    cache --> tx
    cache --> dw

    fn --> models
    sa --> models
    tx --> models
    dw --> models
    dw --> temp
    st --> temp
    st --> data

    fn --> tw
    sa --> tw
    tx --> tw
    tw --> models

    rv -. "live verification" .-> news
    fn -. "translation / remote model fallback" .-> translate
    sa -. "translation" .-> translate
    tx -. "translation" .-> translate
    rv -. "translation" .-> translate
    tw -. "remote transformer download" .-> hf
    dw -. "optional HF deepfake model" .-> hf
```

## Deployment Notes

- The app is deployed as a single primary Streamlit host process.
- Heavy deepfake and transformer inference run in separate subprocess workers so the main UI process remains stable.
- There is no relational database in the current project. Persisted auth data is stored in `users.json`.
- Local model files are loaded from `models/`, while temporary uploads and generated batch artifacts are written under `temp/streamlit_artifacts`.
- External connectivity is optional and is only needed for live news verification, translation, or downloading remote Hugging Face models.

## Basis in Code

- Streamlit app entrypoint and auth flow: `app.py`
- Local user store: `users.json`
- Deepfake subprocess boundary: `safe_deepfake_runtime.py`
- Transformer subprocess boundary: `safe_transformers.py`
- Live verification: `realtime_verifier.py`
- Translation helper: `translator_utils.py`
