def test_key_exchange_oneof():
    from proto import hfedchain_pb2 as pb
    m = pb.ConsensusMessage(key_exchange=pb.KeyExchange(node_id="fog1", vk=b"k"))
    assert m.WhichOneof("msg") == "key_exchange"
    v = pb.ConsensusMessage(vrf_share=pb.VrfShare(node_id="fog1", round=3, y=b"y", proof=b"p"))
    assert v.WhichOneof("msg") == "vrf_share"
    vc = pb.ViewChangeMessage(node_id="fog1", new_view=2, round=7, highest_qc=b"", signature=b"s")
    assert vc.round == 7


def test_field_numbers():
    from proto import hfedchain_pb2 as pb
    kd = pb.KeyExchange.DESCRIPTOR
    assert kd.fields_by_name["node_id"].number == 1
    assert kd.fields_by_name["vk"].number == 2
    vd = pb.VrfShare.DESCRIPTOR
    assert [vd.fields_by_name[n].number for n in ("node_id", "round", "y", "proof")] == [1, 2, 3, 4]
    cm = pb.ConsensusMessage.DESCRIPTOR
    oneof = cm.oneofs_by_name["msg"]
    names = [f.name for f in oneof.fields]
    assert names == [
        "prepare", "vote", "qc", "view_change", "new_view", "key_exchange", "vrf_share",
    ]
    assert cm.fields_by_name["key_exchange"].number == 6
    assert cm.fields_by_name["vrf_share"].number == 7
    vc = pb.ViewChangeMessage.DESCRIPTOR
    assert vc.fields_by_name["round"].number == 5


def test_serialize_roundtrip_and_oneof_exclusivity():
    from proto import hfedchain_pb2 as pb
    m = pb.ConsensusMessage(key_exchange=pb.KeyExchange(node_id="fog1", vk=b"k"))
    assert m.WhichOneof("msg") == "key_exchange"
    m.vrf_share.node_id = "fog1"
    m.vrf_share.round = 3
    m.vrf_share.y = b"y"
    m.vrf_share.proof = b"p"
    assert m.WhichOneof("msg") == "vrf_share"  # setting oneof member replaces the other
    data = m.SerializeToString()
    m2 = pb.ConsensusMessage.FromString(data)
    assert m2.WhichOneof("msg") == "vrf_share"
    assert m2.vrf_share.node_id == "fog1"
    assert m2.vrf_share.round == 3
    assert m2.vrf_share.y == b"y"
    assert m2.vrf_share.proof == b"p"
    assert m2.key_exchange.node_id == ""  # cleared by oneof replacement
