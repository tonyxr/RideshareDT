from scripts import eval_protocol_v1 as protocol


def test_registered_contract_is_internally_consistent():
    spec = protocol.protocol()
    protocol.validate(spec)
    assert len(protocol.CASES) == 6
    assert len(protocol.COMPETITORS) == 10
    assert len(protocol.POLICY_CLASSES) == 10
    assert spec["step_B"]["rollouts_per_city"] == 3000
    assert spec["step_C"]["rollouts_total"] == 1500


def test_protocol_hash_is_stable_and_content_addressed():
    first = protocol.protocol()
    second = protocol.protocol()
    assert protocol.protocol_hash(first) == protocol.protocol_hash(second)
    second["step_B"]["horizon_days"] = 41
    assert protocol.protocol_hash(first) != protocol.protocol_hash(second)


def test_execution_is_deliberately_disabled_in_contract_patch():
    spec = protocol.protocol()
    assert spec["entry_point"] == "scripts/eval_protocol_v1.py"
    assert spec["held_out_tapes"]["must_differ_from"] == [
        "training",
        "T2_validation",
    ]
