import streamlit as st
from pathlib import Path
from llm_engine import process_peer_review, extract_text_from_pdf
import shutil

# ============================================================================
# Demo Files Configuration
# ============================================================================
DEMO_MANUSCRIPT = "BurnRAG_draft.pdf"
DEMO_CHECKLIST = "peer_review-checklist.pdf"
DEMO_FOLDER = Path("demo")
TEMP_FOLDER = Path("temp_uploads")

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
        TEMP_FOLDER.mkdir(exist_ok=True)
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

    # Sidebar - File Upload
    with st.sidebar:
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
                    st.session_state.manuscript_path = str(
                        TEMP_FOLDER / file_name
                    )
                    st.success("Demo manuscript loaded!")
                    st.rerun()

        with col2:
            if st.button(
                "✅ Demo Checklist", use_container_width=True, key="demo_checklist"
            ):
                file_bytes, file_name = load_demo_file(
                    DEMO_CHECKLIST, DEMO_CHECKLIST
                )
                if file_bytes:
                    st.session_state.checklist_file = file_bytes
                    st.session_state.checklist_path = str(
                        TEMP_FOLDER / file_name
                    )
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
                DEMO_CHECKLIST, DEMO_CHECKLIST,
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
                uploaded_manuscript.name
                if uploaded_manuscript
                else DEMO_MANUSCRIPT
            )
            checklist_name = (
                uploaded_checklist.name
                if uploaded_checklist
                else DEMO_CHECKLIST
            )

            if st.button("🚀 Start Review", use_container_width=True, type="primary"):
                with st.spinner("Processing peer review..."):
                    try:
                        TEMP_FOLDER.mkdir(exist_ok=True)

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
                        result = process_peer_review(
                            manuscript_path=st.session_state.manuscript_path,
                            checklist_path=st.session_state.checklist_path,
                            manuscript_name=manuscript_name,
                            checklist_name=checklist_name,
                        )

                        st.session_state.last_result = result

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
