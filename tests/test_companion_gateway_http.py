"""Companion pairing must advertise only a gateway that actually exists."""

import unittest

from fastapi.testclient import TestClient

from api.jarvis_local_server import LocalBrainHost, create_local_brain_app


class CompanionGatewayHttpTests(unittest.TestCase):
    def test_manual_endpoint_cannot_bypass_missing_gateway(self):
        host = object.__new__(LocalBrainHost)
        host.ui_token = "x" * 32
        host.companion_endpoint = None

        app = create_local_brain_app(host)
        client = TestClient(app)
        response = client.post(
            "/v1/pair/offer",
            headers={"Authorization": "Bearer " + host.ui_token},
            json={
                "endpoint": "http://100.64.0.10:8765",
                "ttl_seconds": 300,
            },
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("companion gateway is unavailable", response.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
