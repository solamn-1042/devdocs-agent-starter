"""Day7：上传文件的名称、格式、大小和UTF-8内容校验。"""
import re
from pathlib import Path, PurePosixPath

# 当前知识库加载器只支持 Markdown 和 TXT。
ALLOWED_SUFFIXES = {".md", ".txt"}

# 本地演示限制单个文件最多 2MB，避免一次写入过大的内容。
MAX_UPLOAD_BYTES = 2 * 1024 * 1024


def safe_upload_name(original_name: str) -> str:
    """只保留文件名本身，并清理不适合写入知识库的字符。"""
    # 统一分隔符后只取最后一级名称，不信任浏览器传来的路径。
    normalized = str(original_name or "").replace("\\", "/")
    base_name = PurePosixPath(normalized).name.strip()

    if not base_name or base_name in {".", ".."}:
        raise ValueError("文件名为空或无效。")

    # 后缀统一转成小写，和 retriever.py 的格式判断保持一致。
    suffix = Path(base_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("只允许上传 .md 或 .txt 文件。")

    # 文件名主体只保留中英文、数字、点、下划线和连字符。
    raw_stem = Path(base_name).stem
    safe_stem = re.sub(
        r"[^0-9A-Za-z\u4e00-\u9fff._-]+",
        "_",
        raw_stem,
    ).strip("._-")

    if not safe_stem:
        safe_stem = "uploaded_document"

    return f"{safe_stem}{suffix}"


def decode_uploaded_text(data: bytes, max_bytes: int = MAX_UPLOAD_BYTES) -> str:
    """检查文件大小，并按 UTF-8/UTF-8 BOM 解码为普通文本。"""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("上传内容必须是 bytes。")

    if not data:
        raise ValueError("文件内容为空。")

    if len(data) > max_bytes:
        raise ValueError(f"文件超过大小限制：最多 {max_bytes} 字节。")

    try:
        # utf-8-sig 同时兼容普通 UTF-8 和带 BOM 的 UTF-8。
        content = bytes(data).decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("文件不是有效的 UTF-8 文本，请另存为 UTF-8 后再上传。") from exc

    if not content.strip():
        raise ValueError("文件只有空白内容，不能加入知识库。")

    return content