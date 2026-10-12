"""
test_mvp_suite.py
=================
Comprehensive Automated Test Suite for RailSync AI MVP.
Tests:
- Database connectivity & queries (db_adapter)
- Station-by-station delay prediction engine (prediction/predict.py)
- Railway Network Context Engine (prediction/network_context_service.py)
- Connection Risk Service (prediction/connection_risk_service.py)
- Flask API Endpoints (app.py: /api/trains, /api/stations, /api/rank-indirect, /api/analyse)
"""

import os
import sys
import unittest
import json
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_adapter import get_db_connection, get_all_trains_from_db, get_train_route_from_db, get_train_metadata_from_db
from prediction.predict import predict_train_delay
from prediction.network_context_service import get_network_context
from prediction.connection_risk_service import calculate_connection_risk
from app import app


class TestDatabaseAdapter(unittest.TestCase):
    """Test PostgreSQL database adapter layer."""

    def test_01_db_connection(self):
        conn = get_db_connection()
        self.assertIsNotNone(conn, "PostgreSQL database connection failed.")
        conn.close()

    def test_02_get_all_trains(self):
        trains = get_all_trains_from_db(search_term="11020", limit=10)
        self.assertGreater(len(trains), 0, "Failed to fetch train 11020 from PostgreSQL.")
        t = trains[0]
        self.assertEqual(t["train_number"], "11020")
        self.assertIn("avg_delay_min", t)

    def test_03_get_train_route(self):
        route = get_train_route_from_db("11020")
        self.assertGreater(len(route), 10, "Failed to fetch route for train 11020.")
        self.assertEqual(route[0]["sequence"], 1)

    def test_04_get_train_metadata(self):
        meta = get_train_metadata_from_db("11058")
        self.assertIsNotNone(meta, "Failed to fetch metadata for train 11058.")
        self.assertIn("name", meta)


class TestPredictionEngine(unittest.TestCase):
    """Test XGBoost prediction engine & 28-feature station differentiation."""

    def test_01_predict_train_delay_qualifying_train(self):
        res = predict_train_delay("11020", "BBS", 15.0, "2026-10-10")
        self.assertNotIn("error", res, f"Prediction error: {res.get('error')}")
        self.assertEqual(res["train_number"], "11020")
        self.assertGreater(len(res["forecast"]), 0)

        # Check non-flat delay predictions
        delays = [f["predicted_delay"] for f in res["forecast"]]
        self.assertGreater(len(set(delays)), 1, "Predicted delays across stations should not be flat.")

    def test_02_predict_train_delay_another_train(self):
        res = predict_train_delay("11058", "ASR", 10.0, "2026-10-10")
        self.assertNotIn("error", res)
        self.assertGreater(len(res["forecast"]), 0)


class TestNetworkContextService(unittest.TestCase):
    """Test Railway Network Context Engine."""

    def test_01_get_network_context(self):
        net = get_network_context("11020", "BBS", "CSTM")
        self.assertNotIn("error", net)
        self.assertEqual(net["train_number"], "11020")
        self.assertGreater(len(net["route_segment"]), 0)
        self.assertIn("related_trains", net)
        self.assertIn("network_stats", net)
        self.assertIn("data_provenance", net)


class TestConnectionRiskService(unittest.TestCase):
    """Test connection risk & transfer buffer evaluation."""

    def test_01_safe_connection(self):
        res = calculate_connection_risk("11020", "12801", "BBS", 0.0, "11:00", "12:00")
        self.assertEqual(res["risk_level"], "SAFE")
        self.assertGreaterEqual(res["effective_buffer_min"], 30.0)

    def test_02_uncertain_connection(self):
        res = calculate_connection_risk("11020", "12801", "BBS", 40.0, "11:00", "12:00")
        self.assertEqual(res["risk_level"], "UNCERTAIN")
        self.assertGreaterEqual(res["effective_buffer_min"], 10.0)
        self.assertLess(res["effective_buffer_min"], 30.0)

    def test_03_risky_connection(self):
        res = calculate_connection_risk("11020", "12801", "BBS", 55.0, "11:00", "12:00")
        self.assertEqual(res["risk_level"], "RISKY")
        self.assertLess(res["effective_buffer_min"], 10.0)


class TestFlaskEndpoints(unittest.TestCase):
    """Test Flask web app API endpoints."""

    def setUp(self):
        self.client = app.test_client()

    def test_01_api_trains(self):
        res = self.client.get("/api/trains?search=11020")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertGreater(data.get("total", 0), 0)

    def test_02_api_stations(self):
        res = self.client.get("/api/stations")
        self.assertEqual(res.status_code, 200)

    def test_03_api_rank_indirect(self):
        res = self.client.post("/api/rank-indirect", json={
            "origin": "BBS",
            "destination": "CSMT",
            "initial_delay_min": 15,
            "risk_weight": 0.5,
            "journey_date": "2026-10-10"
        })
        self.assertEqual(res.status_code, 200)

    def test_04_api_analyse(self):
        res = self.client.post("/api/analyse", json={
            "train_number": "11020",
            "source_station": "BBS",
            "dest_station": "CSTM",
            "current_station": "BBS",
            "current_delay": 15.0,
            "journey_date": "2026-10-10",
            "outgoing_train": "12801"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertIsNotNone(data.get("predictions"))
        self.assertIsNotNone(data.get("network_context"))
        self.assertIsNotNone(data.get("connection_risk"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
