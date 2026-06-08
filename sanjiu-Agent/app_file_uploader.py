from __future__ import annotations

import subprocess
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

import config
from knowledge_base import KnowledgeBaseService
from storage import load_json, save_json, now_iso


def _convert_heic_to_png_bytes(image_bytes: bytes) -> bytes | None:
    """Use macOS sips as a fallback converter for HEIC/HEIF images."""
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "upload.heic"
            dst_path = tmp_path / "upload.png"
            src_path.write_bytes(image_bytes)
            subprocess.run(
                ["sips", "-s", "format", "png", str(src_path), "--out", str(dst_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            return dst_path.read_bytes()
    except Exception:
        return None


def _get_preview_image_bytes(uploaded_file) -> bytes | None:
    image_bytes = uploaded_file.getvalue()
    if not image_bytes:
        return None

    try:
        with Image.open(uploaded_file) as img:
            img.verify()
        uploaded_file.seek(0)
        return image_bytes
    except (UnidentifiedImageError, OSError):
        uploaded_file.seek(0)
        suffix = Path(uploaded_file.name or "").suffix.lower()
        if suffix in {".heic", ".heif"}:
            return _convert_heic_to_png_bytes(image_bytes)
        return None


def _preview_uploaded_image(uploaded_file) -> None:
    """Preview image safely so unsupported or broken files don't crash the page."""
    if uploaded_file is None or not uploaded_file.type.startswith("image/"):
        return

    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes:
        st.warning("上传的图片内容为空，请重新选择图片。")
        return

    preview_bytes = _get_preview_image_bytes(uploaded_file)
    if preview_bytes:
        st.image(preview_bytes, caption="已上传药品图片", use_container_width=True)
        uploaded_file.seek(0)
        return

    st.warning(
        "当前图片无法预览，可能是文件损坏或当前环境暂不支持该编码格式。"
        "你仍然可以继续手动补充 OCR 文本和药品信息。"
    )
    uploaded_file.seek(0)


def render_medication_uploader() -> None:
    st.subheader("1. 拍照录药")
    st.caption("支持上传药盒照片或说明书截图；Demo 版提供 OCR 文本补充和人工修正。")

    uploaded_file = st.file_uploader(
        "上传药盒照片 / 说明书截图",
        type=["png", "jpg", "jpeg", "webp", "heic", "heif", "txt"],
        accept_multiple_files=False,
        key="medication_image",
    )

    sample_name = st.selectbox("或使用样例药品快速演示", [""] + list(config.SAMPLE_OCR_TEXT.keys()))
    default_text = config.SAMPLE_OCR_TEXT.get(sample_name, "")
    initial_hint = uploaded_file.name if uploaded_file else ""
    ocr_text = st.text_area(
        "OCR 识别文本（可编辑）",
        value=default_text or initial_hint,
        height=120,
        help="正式版本可接入 PaddleOCR / 通义 OCR；当前 Demo 用可编辑文本模拟识别结果。",
    )

    _preview_uploaded_image(uploaded_file)

    kb_service = KnowledgeBaseService()
    match = kb_service.match_medication_from_text(ocr_text)

    with st.form("medication_confirm_form", clear_on_submit=False):
        st.subheader("2. 信息确认")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("药品名称", value=match.name if match else "")
            spec = st.text_input("规格", value=match.spec if match else "")
            frequency = st.selectbox(
                "服药频次",
                list(config.DEFAULT_FREQUENCY_MAP.keys()),
                index=list(config.DEFAULT_FREQUENCY_MAP.keys()).index(match.frequency) if match else 0,
            )
        with col2:
            times_default = "、".join(match.times) if match else "08:00"
            times = st.text_input("提醒时间（用顿号分隔）", value=times_default)
            usage = st.text_area("用法用量", value=match.usage if match else "", height=100)
            confidence = match.confidence if match else 0.0
            st.metric("识别置信度", f"{confidence:.0%}")

        submitted = st.form_submit_button("加入电子药箱并生成计划", use_container_width=True)

    if submitted:
        medication_box = load_json(config.MEDICATION_BOX_PATH, default=[])
        normalized_name = kb_service.normalize_medication_name(name or ocr_text)
        normalized_usage = usage.strip()
        if not normalized_usage and match:
            normalized_usage = match.usage
        medication_box.append(
            {
                "id": str(uuid.uuid4()),
                "name": normalized_name or config.DEFAULT_MEDICATION_NAME,
                "spec": spec.strip(),
                "frequency": frequency,
                "times": [item.strip() for item in times.split("、") if item.strip()],
                "usage": normalized_usage,
                "ocr_text": ocr_text.strip(),
                "source_file": uploaded_file.name if uploaded_file else sample_name or "manual-input",
                "created_at": now_iso(),
            }
        )
        save_json(config.MEDICATION_BOX_PATH, medication_box)
        st.success("药品已加入电子药箱，并生成基础服药计划。")
        st.rerun()
