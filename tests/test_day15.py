"""Day15：只测HTTP封装，mock掉Day14，不调用模型。"""
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api_day15 import app

class ApiTests(unittest.TestCase):
    def setUp(self):
        """每个测试创建独立HTTP测试客户端。"""
        self.client = TestClient(app)

    def test_health(self):
        """健康检查只证明Web进程在运行。"""
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    @patch("api_day15.safe_turn")
    def test_chat_success(self, mock_turn):
        """验证HTTP字段透传和历史返回。"""
        mock_turn.return_value = {
            "status": "ok", "answer": "请同步索引",
            "history": [{"question": "索引怎么更新？", "answer": "请同步索引"}],
            "trace": [{"trace_id": "mock123", "status": "ok"}]
        }
        r = self.client.post("/chat", json={"question": "索引怎么更新？"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["answer"], "请同步索引")
        mock_turn.assert_called_once_with("索引怎么更新？", [])

    def test_missing_question(self):
        """缺少必填字段应返回422。"""
        self.assertEqual(self.client.post("/chat", json={}).status_code, 422)

    def test_empty_question(self):
        """空字符串不应进入业务模块。"""
        self.assertEqual(self.client.post(
            "/chat", json={"question": ""}).status_code, 422)

    @patch("api_day15.safe_turn")
    def test_service_failure(self, mock_turn):
        """底层失败转换成503，并返回稳定错误类型。"""
        mock_turn.return_value = {
            "status": "failed", "answer": "失败", "history": [],
            "trace": [{"error": "timeout"}]
        }
        r = self.client.post("/chat", json={"question": "超时示例"})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["detail"]["error"], "timeout")

    def test_invalid_history(self):
        """历史元素结构不合法时由Pydantic拒绝。"""
        r = self.client.post("/chat", json={
            "question": "继续", "history": [{"question": "旧问题"}]})
        self.assertEqual(r.status_code, 422)

if __name__ == "__main__":
    unittest.main()