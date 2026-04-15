"""Streamlit interface for batch patent mapping across uploaded files or local folders."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os

import streamlit as st
from dotenv import load_dotenv

from utils.agent_builder import PROVIDER_MODELS, build_patent_mapping_chain, get_provider_models
from utils.export_utils import safe_filename, write_analysis_file
from utils.extractors import DEFAULT_OUTPUT_INSTRUCTIONS, format_key_features, parse_key_features
from utils.pdf_processor import build_document_artifact, build_document_artifact_from_path, iter_supported_files

load_dotenv()

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = APP_ROOT / "analysis_results"

st.set_page_config(page_title="PDF Patent Mapper", page_icon="P", layout="wide")

st.markdown(
    """
<style>
    .hero {
        padding: 1rem 0 0.25rem 0;
    }
    .hero h1 {
        margin-bottom: 0.25rem;
    }
    .panel {
        border: 1px solid rgba(120,120,120,0.25);
        border-radius: 16px;
        padding: 1rem;
        background: rgba(255,255,255,0.03);
    }
    .muted {
        color: #6b7280;
        font-size: 0.95rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

DEFAULTS = {
    "batch_results": [],
    "batch_output_dir": None,
    "last_error": "",
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def unique_output_path(directory: Path, stem: str, suffix: str = ".txt") -> Path:
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate

    counter = 2
    while True:
        numbered = directory / f"{stem}_{counter}{suffix}"
        if not numbered.exists():
            return numbered
        counter += 1


def normalize_analysis_tag(tag: str) -> str:
    cleaned = safe_filename(tag or "analysis")
    return cleaned or f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _resolve_from_project(project_path: Path, value: str, default_child: str) -> Path:
    raw = (value or "").strip()
    if not raw:
        return project_path / default_child
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else project_path / candidate


def resolve_folder_paths(project_folder: str, source_folder: str, results_folder: str) -> tuple[str, str]:
    if project_folder.strip():
        project_path = Path(project_folder).expanduser()
        derived_source = _resolve_from_project(project_path, source_folder, "source")
        derived_results = _resolve_from_project(project_path, results_folder, "results")
        return str(derived_source), str(derived_results)

    source_raw = (source_folder or "").strip()
    results_raw = (results_folder or "").strip()
    source_path = Path(source_raw).expanduser() if source_raw else Path("")
    results_path = Path(results_raw).expanduser() if results_raw else DEFAULT_RESULTS_DIR

    if source_raw and not source_path.is_absolute():
        source_path = APP_ROOT / source_path
    if results_raw and not results_path.is_absolute():
        results_path = APP_ROOT / results_path

    return str(source_path) if source_raw else "", str(results_path)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_writable_directory(path: Path) -> None:
    ensure_directory(path)
    probe = path / ".write_probe.tmp"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)


def is_windows_path(path_str: str) -> bool:
    value = (path_str or "").strip()
    return len(value) > 1 and value[1] == ":"


def collect_sources(uploaded_files, input_folder: str, output_folder: str, enable_ocr: bool, chunk_size: int, chunk_overlap: int, provider: str):
    sources = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            sources.append(
                {
                    "kind": "upload",
                    "name": uploaded_file.name,
                    "artifact": build_document_artifact(
                        uploaded_file,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        provider=provider,
                        enable_ocr=enable_ocr,
                    ),
                }
            )

    folder = (input_folder or "").strip()
    if folder:
        input_path = Path(folder).expanduser()
        if not input_path.exists():
            input_path.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_folder).expanduser() if output_folder else None
        folder_files = iter_supported_files(input_path)
        for file_path in folder_files:
            if output_path is not None:
                try:
                    resolved_output = output_path.resolve()
                    if resolved_output in file_path.resolve().parents:
                        continue
                except Exception:
                    pass
            stem_lower = file_path.stem.lower()
            if "_map_" in stem_lower or stem_lower.startswith("ocr_candidates") or stem_lower.startswith("analysis_manifest"):
                continue
            sources.append(
                {
                    "kind": "folder",
                    "name": file_path.name,
                    "path": file_path,
                    "artifact": build_document_artifact_from_path(
                        file_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        provider=provider,
                        enable_ocr=enable_ocr,
                    ),
                }
            )

    return sources


with st.sidebar:
    st.markdown("## Patent Mapping Settings")
    provider = st.selectbox("LLM provider", list(PROVIDER_MODELS.keys()), index=2)
    model_options = get_provider_models(provider)
    model_name = st.selectbox("Model", model_options, index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    st.divider()
    st.markdown("### Batch controls")
    chunk_size = st.slider("Chunk size", 400, 3000, 1200, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 600, 200, 50)
    enable_ocr = st.checkbox("Enable OCR for image-heavy PDF pages", value=True)
    analysis_tag = st.text_input("Analysis tag", value=safe_filename(model_name))
    quality_mode = st.selectbox("Mapping quality mode", ["High", "Fast"], index=0)
    retrieval_k = st.slider("Evidence chunks per feature", 3, 12, 8 if quality_mode == "High" else 5)
    output_format = st.selectbox("Output format", ["txt", "docx", "pdf"], index=0)
    st.caption("The report name becomes `document_name_Map_analysis-tag.txt`.")

    st.divider()
    st.markdown("### Folders")
    project_folder = st.text_input("Project folder (optional)", value="")
    source_folder = st.text_input("Source folder path", value="")
    results_folder = st.text_input("Results folder path", value="")
    st.caption("If you provide a project folder, the app uses `project/source` and `project/results` by default.")

    st.divider()
    if st.button("Create Source/Results Folders", use_container_width=True):
        try:
            source_tmp, results_tmp = resolve_folder_paths(project_folder, source_folder, results_folder)
            if source_tmp.strip():
                ensure_directory(Path(source_tmp).expanduser())
            if results_tmp.strip():
                ensure_directory(Path(results_tmp).expanduser())
            st.success("Folder structure is ready.")
        except Exception as exc:
            st.error(f"Folder creation failed: {exc}")

    st.divider()
    st.caption("Local Ollama models are refreshed automatically from your server. Use `ollama pull <model>` to update them offline.")


st.markdown('<div class="hero"><h1>PDF Patent Mapper</h1></div>', unsafe_allow_html=True)
st.write(
    "Upload up to 300 documents or point the app at a folder, provide the key features file, and generate one analysis file per document in your selected output format."
)

col_left, col_right = st.columns([1.3, 1])
with col_left:
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="visible",
    )
    st.caption("Accepted formats: PDF, DOCX, TXT, and MD. You can use uploads, a local folder, or both.")

with col_right:
    key_features_file = st.file_uploader("Upload key features file", type=["txt", "md"], accept_multiple_files=False)
    output_instructions_file = st.file_uploader("Upload output instructions file", type=["txt", "md"], accept_multiple_files=False)
    st.caption("If you do not upload the instructions file, the default mapping format from your sample is used.")

st.subheader("Key features")
key_features_text = st.text_area(
    "Paste the numbered key features here if you do not want to upload a file.",
    height=260,
    placeholder="Paste numbered key features such as 1, 1.1, 1.2 ...",
)

st.subheader("Output instructions")
output_instructions_text = st.text_area(
    "Paste the output instructions here if needed.",
    value=DEFAULT_OUTPUT_INSTRUCTIONS,
    height=220,
)

if key_features_file:
    key_features_text = key_features_file.getvalue().decode("utf-8", errors="ignore")
if output_instructions_file:
    output_instructions_text = output_instructions_file.getvalue().decode("utf-8", errors="ignore")

features = parse_key_features(key_features_text)

source_folder_value, results_folder_value = resolve_folder_paths(project_folder, source_folder, results_folder)

if os.name != "nt":
    if is_windows_path(source_folder_value) or is_windows_path(results_folder_value):
        st.warning("Windows-style folder paths were detected while running on a non-Windows OS. Use container-mounted Linux paths when running inside Docker.")

source_preview_count = len(uploaded_files or [])
folder_count = 0
if source_folder_value.strip():
    try:
        folder_count = len(
            [
                p
                for p in iter_supported_files(source_folder_value)
                if "_map_" not in p.stem.lower()
                and not p.stem.lower().startswith("ocr_candidates")
                and not p.stem.lower().startswith("analysis_manifest")
            ]
        )
    except Exception:
        folder_count = 0

info_cols = st.columns(5)
info_cols[0].metric("Uploaded files", source_preview_count)
info_cols[1].metric("Folder files", folder_count)
info_cols[2].metric("Parsed features", len(features))
info_cols[3].metric("Chunk size", chunk_size)
info_cols[4].metric("OCR", "On" if enable_ocr else "Off")

st.caption("Resolved source folder")
st.code(source_folder_value or "(using upload-only mode)", language="text")
st.caption("Resolved results folder")
st.code(results_folder_value, language="text")

if features:
    with st.expander("Preview parsed features", expanded=False):
        st.code(format_key_features(features), language="text")

action_cols = st.columns(3)
run_clicked = action_cols[0].button("Run mapping batch", type="primary", use_container_width=True)
clear_clicked = action_cols[1].button("Clear results", use_container_width=True)
reset_clicked = action_cols[2].button("Reset app", use_container_width=True)

if clear_clicked:
    st.session_state.batch_results = []
    st.session_state.batch_output_dir = None
    st.session_state.last_error = ""
    st.rerun()

if reset_clicked:
    for session_key in list(st.session_state.keys()):
        del st.session_state[session_key]
    st.rerun()


def _read_output_file(path: str) -> bytes:
    return Path(path).read_bytes()


def _output_mime(format_name: str) -> str:
    fmt = (format_name or "").lower()
    if fmt == "pdf":
        return "application/pdf"
    if fmt == "docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return "text/plain"


if run_clicked:
    if not features:
        st.error("Please provide a numbered key features file or paste the key features text.")
        st.stop()

    if not uploaded_files and not source_folder_value.strip():
        st.error("Please upload documents, provide an input folder, or use both.")
        st.stop()

    try:
        output_dir = Path(results_folder_value).expanduser()
        ensure_writable_directory(output_dir)

        if source_folder_value.strip():
            source_dir = Path(source_folder_value).expanduser()
            ensure_directory(source_dir)

        sources = collect_sources(
            uploaded_files=uploaded_files,
            input_folder=source_folder_value,
            output_folder=output_dir,
            enable_ocr=enable_ocr,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            provider=provider,
        )

        if not sources:
            st.error(f"No supported documents were found to process. Check source folder: {source_folder_value or '(upload-only)'}")
            st.stop()

        if len(sources) > 300:
            st.warning("Only the first 300 documents will be processed in this run.")
            sources = sources[:300]

        mapper = build_patent_mapping_chain(provider, model_name, temperature)
        run_suffix = normalize_analysis_tag(analysis_tag or model_name)
        selected_suffix = f".{output_format.lower()}"

        report_rows: list[dict] = []
        progress = st.progress(0)
        status_text = st.empty()
        status_text.info("Starting batch processing...")

        total_files = len(sources)
        for index, source in enumerate(sources, start=1):
            artifact = source["artifact"]
            status_text.info(f"Processing {artifact.source_filename} ({index}/{total_files})")

            if not artifact.vector_store:
                raise RuntimeError(f"No searchable content extracted from {artifact.source_filename}")

            feature_contexts = mapper.collect_feature_contexts(artifact.vector_store, features, k=retrieval_k)
            mapping_text = mapper.map_document(
                document_name=artifact.source_filename,
                features=features,
                feature_contexts=feature_contexts,
                output_instructions=output_instructions_text,
                quality_mode=quality_mode.lower(),
            )

            base_name = f"{safe_filename(artifact.source_filename)}_Map_{run_suffix}"
            output_path = unique_output_path(output_dir, base_name, selected_suffix)
            write_analysis_file(output_path, mapping_text, output_format)

            report_rows.append(
                {
                    "name": artifact.source_filename,
                    "output_path": str(output_path),
                    "preview_text": mapping_text[:5000],
                    "output_format": output_format.lower(),
                    "pages": artifact.pages,
                    "chunks": artifact.chunks,
                    "image_pages": artifact.image_pages,
                    "ocr_needed_pages": artifact.ocr_needed_pages,
                    "status": "Completed",
                }
            )

            progress.progress(index / total_files)

        st.session_state.batch_results = report_rows
        st.session_state.batch_output_dir = str(output_dir)
        st.session_state.last_error = ""

        status_text.success(f"Created {len(report_rows)} analysis reports in {output_dir}")
        st.success(f"Created {len(report_rows)} analysis reports in {output_dir}")
    except Exception as exc:
        st.session_state.last_error = str(exc)
        st.error(str(exc))


if st.session_state.last_error:
    st.error(st.session_state.last_error)


if st.session_state.batch_results:
    st.subheader("Batch summary")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Reports", len(st.session_state.batch_results))
    summary_cols[1].metric("Format", output_format.upper())
    summary_cols[2].metric("Output dir", "Ready")
    summary_cols[3].metric("One file", "Per analysis")

    st.markdown("### Per-document reports")
    for row in st.session_state.batch_results:
        with st.expander(f"{row['name']} - {row['status']}", expanded=False):
            st.write(f"Pages: {row['pages']} | Chunks: {row['chunks']}")
            if row.get("ocr_needed_pages"):
                st.write(f"OCR recommended for pages: {', '.join(map(str, row['ocr_needed_pages']))}")
            output_path = Path(row["output_path"])
            if output_path.exists():
                preview = row.get("preview_text", "")
                st.text_area("Preview", value=preview, height=280, key=f"preview_{output_path.stem}")
                st.download_button(
                    f"Download {row.get('output_format', output_format).upper()}",
                    data=output_path.read_bytes(),
                    file_name=output_path.name,
                    mime=_output_mime(row.get("output_format", output_format)),
                    key=f"download_{output_path.stem}",
                )
