"""Day7：上传文件校验的离线测试，不启动 Streamlit 或外部服务。"""
import unittest

from upload_utils import safe_upload_name, decode_uploaded_text


class TestDay7UploadUtils(unittest.TestCase):
    def test_windows_path_only_keeps_file_name(self):
        result = safe_upload_name(r"C:\temp\private\notes.txt")
        self.assertEqual(result, "notes.txt")

    def test_spaces_are_normalized(self):
        result = safe_upload_name("研发 规范.md")
        self.assertEqual(result, "研发_规范.md")

    def test_unsupported_extension_is_rejected(self):
        with self.assertRaises(ValueError):
            safe_upload_name("secret.pdf")

    def test_utf8_bom_is_removed(self):
        result = decode_uploaded_text("# 标题\n正文".encode("utf-8-sig"))
        self.assertEqual(result, "# 标题\n正文")

    def test_invalid_utf8_is_rejected(self):
        with self.assertRaises(ValueError):
            decode_uploaded_text(b"\xff\xfe")

    def test_empty_or_whitespace_file_is_rejected(self):
        with self.assertRaises(ValueError):
            decode_uploaded_text(b"")
        with self.assertRaises(ValueError):
            decode_uploaded_text(b" \r\n ")

    def test_file_over_size_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            decode_uploaded_text(b"12345", max_bytes=4)


if __name__ == "__main__":
    unittest.main()