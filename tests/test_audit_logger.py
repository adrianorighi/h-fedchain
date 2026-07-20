import pytest
from core.audit.logger import AuditLogger


class TestAuditLogger:
    def test_log_and_count(self):
        logger = AuditLogger()
        logger.log("REJ_PKI", "d0", 1)
        logger.log("REJ_ZKP", "d1", 1)
        logger.log("REJ_PKI", "d2", 2)
        assert logger.count_by_type("REJ_PKI") == 2
        assert logger.count_by_type("REJ_ZKP") == 1
        assert logger.count_by_type("VIEW_CHANGE") == 0

    def test_count_by_round(self):
        logger = AuditLogger()
        logger.log("REJ_PKI", "d0", 1)
        logger.log("REJ_ZKP", "d1", 1)
        logger.log("LEADER_ELECTION", "n0", 2)
        assert logger.count_by_round(1) == 2
        assert logger.count_by_round(2) == 1

    def test_to_list(self):
        logger = AuditLogger()
        logger.log("REJ_PKI", "d0", 1, {"reason": "signature"})
        entries = logger.to_list()
        assert len(entries) == 1
        assert entries[0]["event_type"] == "REJ_PKI"
        assert entries[0]["node_id"] == "d0"
        assert entries[0]["metadata"]["reason"] == "signature"

    def test_clear(self):
        logger = AuditLogger()
        logger.log("QC_COMMIT", "n0", 1)
        assert len(logger.get_entries()) == 1
        logger.clear()
        assert len(logger.get_entries()) == 0

    def test_event_types_defined(self):
        assert "REJ_PKI" in AuditLogger.EVENT_TYPES
        assert "REJ_ZKP" in AuditLogger.EVENT_TYPES
        assert "VIEW_CHANGE" in AuditLogger.EVENT_TYPES
