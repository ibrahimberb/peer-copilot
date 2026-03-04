import streamlit as st
from pathlib import Path
from llm_engine import (
    process_peer_review,
    extract_text_from_pdf,
    evaluate_manuscript_with_checklist,
    generate_final_consideration,
)
import shutil
import tempfile
import yaml

# ============================================================================
# Demo Files Configuration
# ============================================================================
DEMO_MANUSCRIPT = "BurnRAG_draft.pdf"
DEMO_CHECKLIST = "peer_review-checklist.pdf"
DEMO_FOLDER = Path("demo")
TEMP_FOLDER = Path(tempfile.gettempdir()) / "peer_copilot"


# ============================================================================
# LLM Configuration Functions
# ============================================================================
def update_llm_config(provider: str, model: str):
    """Update the config.yaml file with new LLM provider settings."""
    config_path = Path("config.yaml")

    try:
        # Load existing config
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Update provider and model
        config["llm"]["provider"] = provider
        config["llm"]["model"] = model

        # Save updated config
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return True
    except Exception as e:
        st.error(f"Failed to update config: {e}")
        return False


def get_current_llm_config():
    """Get current LLM configuration."""
    config_path = Path("config.yaml")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config["llm"]["provider"], config["llm"]["model"]
    except Exception:
        return "lm_studio", "openai/gpt-oss-20b"


# ============================================================================
# Page configuration
st.set_page_config(
    page_title="Peer Copilot - AI Peer Review",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for elegant buttons
st.markdown(
    """
    <style>
    .demo-button {
        background-color: #f0f2f6;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
        transition: all 0.2s;
    }
    .demo-button:hover {
        background-color: #e5e7eb;
        border-color: #9ca3af;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .status-uploaded {
        background-color: #d1fae5;
        color: #065f46;
    }
    .status-empty {
        background-color: #fee2e2;
        color: #991b1b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_demo_file(demo_path: str, file_name: str):
    """Load a demo file from the demo folder."""
    demo_file_path = DEMO_FOLDER / demo_path
    if demo_file_path.exists():
        temp_path = TEMP_FOLDER / file_name

        # Copy demo file to temp_uploads
        shutil.copy2(demo_file_path, temp_path)

        # Read file as bytes to simulate upload
        with open(temp_path, "rb") as f:
            file_bytes = f.read()

        return file_bytes, file_name
    return None, None


def main():
    # Title
    st.title("AI Peer Review Tool")

    # Initialize session state
    if "manuscript_file" not in st.session_state:
        st.session_state.manuscript_file = None
    if "checklist_file" not in st.session_state:
        st.session_state.checklist_file = None
    if "manuscript_path" not in st.session_state:
        st.session_state.manuscript_path = None
    if "checklist_path" not in st.session_state:
        st.session_state.checklist_path = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = ""
    if "checklist_items" not in st.session_state:
        st.session_state.checklist_items = []
    if "review_results" not in st.session_state:
        st.session_state.review_results = []
    if "evaluation_complete" not in st.session_state:
        st.session_state.evaluation_complete = False
    if "final_consideration" not in st.session_state:
        st.session_state.final_consideration = None
    if "storage_initialized" not in st.session_state:
        st.session_state.storage_initialized = False

    # ── Storage initialization (runs once per session) ──────────────────────
    if not st.session_state.storage_initialized:
        try:
            TEMP_FOLDER.mkdir(parents=True, exist_ok=True)
            init_file = TEMP_FOLDER / "init_check.txt"
            from datetime import datetime
            init_file.write_text(
                f"peer_copilot storage initialized at {datetime.utcnow().isoformat()}Z\n",
                encoding="utf-8",
            )
            st.session_state.storage_initialized = True
            st.success(f"✅ Storage initialized and writable at `{TEMP_FOLDER}`")
            print(f"✅ Storage initialized and writable at `{TEMP_FOLDER}`")
        except Exception as e:
            st.error(f"❌ Storage initialization failed: {e}")

    # Sidebar - File Upload
    with st.sidebar:
        # LLM Provider Selection
        st.header("✨ LLM Provider")

        # Get current config
        current_provider, current_model = get_current_llm_config()

        # Map provider display names to provider keys
        provider_display_map = {
            "Local LLM": "lm_studio",
            "OpenAI": "openai",
        }

        # Map provider to available models
        provider_models = {
            "lm_studio": ["openai/gpt-oss-20b", "llama-3.1-8b", "mistral-7b"],
            "openai": ["gpt-4o-mini", "gpt-5-nano", "gpt-5-mini", "gpt-4o", "gpt-5", "gpt-5.2"],
        }

        # Determine current provider selection
        current_provider_display = "Local LLM"
        for display_name, prov_key in provider_display_map.items():
            if current_provider == prov_key:
                current_provider_display = display_name
                break

        # Provider dropdown
        selected_provider_display = st.selectbox(
            "Select LLM Provider",
            options=list(provider_display_map.keys()),
            index=list(provider_display_map.keys()).index(current_provider_display),
            key="llm_provider_selector",
        )

        provider_key = provider_display_map[selected_provider_display]

        # Model dropdown based on selected provider
        available_models = provider_models[provider_key]
        
        # Ensure current model is in the list, otherwise use first available
        if current_model not in available_models:
            model_index = 0
        else:
            model_index = available_models.index(current_model)

        selected_model = st.selectbox(
            "Select Model",
            options=available_models,
            index=model_index,
            key="llm_model_selector",
        )

        # Update config if selection changed
        if current_provider != provider_key or current_model != selected_model:
            if update_llm_config(provider_key, selected_model):
                st.success(f"✅ Switched to {selected_provider_display} - {selected_model}")
                # Reload the config in llm_engine
                import llm_engine

                llm_engine.config = llm_engine.load_config()

        st.markdown("---")

        st.header("📄 Upload Files")

        # Demo Files Section
        st.subheader("Demo Files")
        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "📄 Demo Manuscript", use_container_width=True, key="demo_manuscript"
            ):
                file_bytes, file_name = load_demo_file(
                    DEMO_MANUSCRIPT,
                    DEMO_MANUSCRIPT,
                )
                if file_bytes:
                    st.session_state.manuscript_file = file_bytes
                    st.session_state.manuscript_path = str(TEMP_FOLDER / file_name)
                    st.success("Demo manuscript loaded!")
                    st.rerun()

        with col2:
            if st.button(
                "✅ Demo Checklist", use_container_width=True, key="demo_checklist"
            ):
                file_bytes, file_name = load_demo_file(DEMO_CHECKLIST, DEMO_CHECKLIST)
                if file_bytes:
                    st.session_state.checklist_file = file_bytes
                    st.session_state.checklist_path = str(TEMP_FOLDER / file_name)
                    st.success("Demo checklist loaded!")
                    st.rerun()

        if st.button(
            "📚 Load Both Demo Files", use_container_width=True, key="demo_both"
        ):
            ms_bytes, ms_name = load_demo_file(
                DEMO_MANUSCRIPT,
                DEMO_MANUSCRIPT,
            )
            cl_bytes, cl_name = load_demo_file(
                DEMO_CHECKLIST,
                DEMO_CHECKLIST,
            )
            if ms_bytes and cl_bytes:
                st.session_state.manuscript_file = ms_bytes
                st.session_state.manuscript_path = str(TEMP_FOLDER / ms_name)
                st.session_state.checklist_file = cl_bytes
                st.session_state.checklist_path = str(TEMP_FOLDER / cl_name)
                st.success("Both demo files loaded!")
                st.rerun()

        st.markdown("---")

        # Manuscript Upload
        st.subheader("Manuscript")
        manuscript_status = (
            "✅ Uploaded" if st.session_state.manuscript_file else "❌ Not uploaded"
        )
        st.markdown(f"**Status:** {manuscript_status}")

        uploaded_manuscript = st.file_uploader(
            "Upload Manuscript (PDF)",
            type=["pdf"],
            key="manuscript_uploader",
        )

        if uploaded_manuscript is not None:
            st.session_state.manuscript_file = uploaded_manuscript.getbuffer()
            st.session_state.manuscript_path = None  # Will be set when processing

        # Checklist Upload
        st.subheader("Review Checklist")
        checklist_status = (
            "✅ Uploaded" if st.session_state.checklist_file else "❌ Not uploaded"
        )
        st.markdown(f"**Status:** {checklist_status}")

        uploaded_checklist = st.file_uploader(
            "Upload Review Checklist (PDF)",
            type=["pdf"],
            key="checklist_uploader",
        )

        if uploaded_checklist is not None:
            st.session_state.checklist_file = uploaded_checklist.getbuffer()
            st.session_state.checklist_path = None  # Will be set when processing

        st.markdown("---")

        # Process Button
        both_files_ready = (
            st.session_state.manuscript_file is not None
            and st.session_state.checklist_file is not None
        )

        if both_files_ready:
            manuscript_name = (
                uploaded_manuscript.name if uploaded_manuscript else DEMO_MANUSCRIPT
            )
            checklist_name = (
                uploaded_checklist.name if uploaded_checklist else DEMO_CHECKLIST
            )

            if st.button("🚀 Start Review", use_container_width=True, type="primary"):
                with st.spinner("Processing peer review..."):
                    try:
                        # Save manuscript
                        if uploaded_manuscript:
                            manuscript_path = TEMP_FOLDER / uploaded_manuscript.name
                            with open(manuscript_path, "wb") as f:
                                f.write(st.session_state.manuscript_file)
                            st.session_state.manuscript_path = str(manuscript_path)
                            manuscript_name = uploaded_manuscript.name
                        else:
                            # Demo file already in temp_uploads
                            manuscript_name = DEMO_MANUSCRIPT
                            st.session_state.manuscript_path = str(
                                TEMP_FOLDER / manuscript_name
                            )

                        # Save checklist
                        if uploaded_checklist:
                            checklist_path = TEMP_FOLDER / uploaded_checklist.name
                            with open(checklist_path, "wb") as f:
                                f.write(st.session_state.checklist_file)
                            st.session_state.checklist_path = str(checklist_path)
                            checklist_name = uploaded_checklist.name
                        else:
                            # Demo file already in temp_uploads
                            checklist_name = DEMO_CHECKLIST
                            st.session_state.checklist_path = str(
                                TEMP_FOLDER / checklist_name
                            )

                        # Process peer review
                        result, checklist_items = process_peer_review(
                            manuscript_path=st.session_state.manuscript_path,
                            checklist_path=st.session_state.checklist_path,
                            manuscript_name=manuscript_name,
                            checklist_name=checklist_name,
                        )

                        st.session_state.last_result = result
                        st.session_state.checklist_items = checklist_items
                        st.session_state.evaluation_complete = False

                        st.success(
                            "✅ Checklist extracted! Now evaluating manuscript..."
                        )

                        # Clean up temp files
                        if (
                            uploaded_manuscript
                            and Path(st.session_state.manuscript_path).exists()
                        ):
                            Path(st.session_state.manuscript_path).unlink()
                        if (
                            uploaded_checklist
                            and Path(st.session_state.checklist_path).exists()
                        ):
                            Path(st.session_state.checklist_path).unlink()

                        st.success("Review completed!")
                        st.rerun()

                    except Exception as e:
                        error_msg = f"Error processing review: {str(e)}"
                        st.error(error_msg)
                        # Clean up on error
                        if (
                            st.session_state.manuscript_path
                            and Path(st.session_state.manuscript_path).exists()
                        ):
                            Path(st.session_state.manuscript_path).unlink()
                        if (
                            st.session_state.checklist_path
                            and Path(st.session_state.checklist_path).exists()
                        ):
                            Path(st.session_state.checklist_path).unlink()
        else:
            st.info("👆 Please upload both files to start the review.")

    # Main content - Review Results
    st.header("Review Results")

    if st.session_state.last_result:
        st.markdown(st.session_state.last_result)

        # Show extracted checklist items (ALWAYS visible after extraction)
        if st.session_state.checklist_items:
            st.markdown("---")
            st.subheader("📋 Structured Checklist Items (Interactive)")

            with st.expander(
                f"View {len(st.session_state.checklist_items)} Checklist Items",
                expanded=True,
            ):
                # Group by section
                sections = {}
                for item in st.session_state.checklist_items:
                    section = item["section"]
                    if section not in sections:
                        sections[section] = []
                    sections[section].append(item)

                # Display by section
                for section, items in sections.items():
                    st.markdown(f"**{section}**")
                    for idx, item in enumerate(items):
                        indent = "  " if item["type"] == "sub" else ""
                        icon = "❓" if item["is_question"] else "📌"
                        st.markdown(f"{indent}{icon} {item['item']}")
                    st.markdown("")

    # Auto-trigger evaluation BELOW the checklist display
    if (
        st.session_state.checklist_items
        and not st.session_state.evaluation_complete
        and st.session_state.manuscript_path
        and st.session_state.checklist_path
    ):

        st.markdown("---")
        st.header("🤖 Evaluating Manuscript")

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        current_item_display = st.empty()

        total_items = len(st.session_state.checklist_items)

        def update_progress(current, total, item_name):
            progress = current / total
            progress_bar.progress(progress)
            status_text.markdown(f"**Processing item {current} of {total}**")
            current_item_display.info(f"💭 Thinking about: *{item_name}*")

        # Read manuscript text
        manuscript_text = extract_text_from_pdf(st.session_state.manuscript_path)

        # Evaluate with checklist
        review_results_path = TEMP_FOLDER / "review_results.json"
        review_results = evaluate_manuscript_with_checklist(
            checklist_items=st.session_state.checklist_items,
            manuscript_text=manuscript_text,
            output_path=str(review_results_path),
            progress_callback=update_progress,
        )

        # Store results
        st.session_state.review_results = review_results
        st.session_state.evaluation_complete = True

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        current_item_display.empty()

        st.success(f"✅ Evaluation complete! Assessed {total_items} checklist items.")
        st.rerun()

    # Display evaluation results if available
    if st.session_state.review_results and st.session_state.evaluation_complete:
        st.markdown("---")
        st.subheader("📊 Detailed Evaluation Results")

        # Group results by section
        sections = {}
        for result in st.session_state.review_results:
            section = result["section"]
            if section not in sections:
                sections[section] = []
            sections[section].append(result)

        # Display by section with expandable cards
        for section, items in sections.items():
            with st.expander(f"📁 {section} ({len(items)} items)", expanded=False):
                for result in items:
                    item_icon = "❓" if result["is_question"] else "📌"
                    status_icon = "✅" if result["status"] == "completed" else "❌"

                    st.markdown(f"**{item_icon} {result['checklist_item']}**")

                    # Display LLM evaluation
                    if result["status"] == "completed":
                        st.markdown(f"**Evaluation:** {result['llm_evaluation']}")
                    else:
                        st.error(f"Error: {result['llm_evaluation']}")

                    st.markdown("---")

        # Generate and display final consideration
        if st.session_state.final_consideration is None:
            st.markdown("---")
            st.header("🤖 Generating Final Consideration")

            with st.spinner(
                "Analyzing all results and generating final recommendation..."
            ):
                # Read manuscript text for context
                manuscript_text = extract_text_from_pdf(
                    st.session_state.manuscript_path
                )

                # Generate final consideration
                final_consideration = generate_final_consideration(
                    review_results=st.session_state.review_results,
                    manuscript_text=manuscript_text,
                )

                st.session_state.final_consideration = final_consideration
                st.rerun()

        # Display final consideration if available
        if st.session_state.final_consideration:
            st.markdown("---")
            st.header("🎯 Final Consideration")

            final = st.session_state.final_consideration

            if final["status"] == "completed":
                # Determine color and emoji based on recommendation
                recommendation = final["recommendation"].upper()
                if recommendation == "ACCEPT":
                    badge_color = "#10b981"  # Green
                    badge_emoji = "✅"
                elif recommendation == "MINOR REVISION":
                    badge_color = "#f59e0b"  # Amber
                    badge_emoji = "⚠️"
                elif recommendation == "MAJOR REVISION":
                    badge_color = "#ef4444"  # Red
                    badge_emoji = "🔴"
                else:  # REJECT
                    badge_color = "#dc2626"  # Dark Red
                    badge_emoji = "❌"

                # Display recommendation with styling
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(
                        f"""<div style="
                            background-color: {badge_color};
                            color: white;
                            padding: 20px;
                            border-radius: 10px;
                            text-align: center;
                            font-size: 20px;
                            font-weight: bold;
                        ">{badge_emoji} {recommendation}</div>""",
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.metric("Confidence Level", f"{final['confidence']}%", delta=None)

                st.markdown("### Reasoning")
                st.info(final["reasoning"])
            else:
                st.error(
                    f"Failed to generate final consideration: {final['reasoning']}"
                )
    else:
        st.info(
            "Upload manuscript and checklist files, then click 'Start Review' to see results here."
        )

    # Text area for detailed results (reserved for future structured display)
    st.markdown("---")
    st.subheader("Detailed Output")

    display_text = st.session_state.last_result if st.session_state.last_result else ""

    st.text_area(
        "Review Output",
        value=display_text,
        height=400,
        placeholder="Structured review results will appear here...",
        key="review_display",
        label_visibility="collapsed",
    )


if __name__ == "__main__":
    main()
