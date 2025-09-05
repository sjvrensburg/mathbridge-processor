from mathbridge_processor.data_cleaner import MathBridgeCleaner


def test_clean_record_basic():
    cleaner = MathBridgeCleaner()
    rec = {"spoken_English": "this is a test.", "equation": "  x  +  y  =  z  "}
    new_rec, counts = cleaner.clean_record(rec)
    assert new_rec["spoken_English"].endswith("test")
    assert new_rec["spoken_English"][0].isupper()
    assert new_rec["equation"] == "x + y = z"
    assert counts["removed_periods"] == 1
    assert counts["normalized_whitespace"] == 1


def test_clean_batch_retains_keys():
    cleaner = MathBridgeCleaner()
    recs = [
        {"context_before": "A", "equation": "a  +  b", "context_after": "C", "spoken_English": "sum."},
    ]
    cleaned, counts = cleaner.clean_batch(recs)
    assert cleaned[0]["context_before"] == "A"
    assert cleaned[0]["context_after"] == "C"
    assert "sre_spoken_text" not in cleaned[0]
    assert counts["removed_periods"] == 1
