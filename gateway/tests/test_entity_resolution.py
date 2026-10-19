import random
import string

from services.entity_resolution import _similarity, candidate_pairs


def test_duplicates_share_a_block():
    rows = [
        {"customer_id": 1, "first_name": "Jonathan", "last_name": "Smith", "email": "jon@x.com"},
        {"customer_id": 2, "first_name": "Jonathon", "last_name": "Smith", "email": "j.smith@y.com"},
        {"customer_id": 3, "first_name": "Maria", "last_name": "Garcia", "email": "jon@x.com"},
    ]
    pairs = candidate_pairs(rows, "email", ["first_name", "last_name"])
    assert (0, 1) in pairs  # name prefix/suffix
    assert (0, 2) in pairs  # same email


def test_typo_at_start_still_blocked_by_suffix():
    rows = [{"name": "Xmartwatch"}, {"name": "Smartwatch"}]
    assert (0, 1) in candidate_pairs(rows, None, ["name"])


def test_blocking_prunes_most_pairs():
    random.seed(0)
    rows = [
        {"name": "".join(random.choices(string.ascii_lowercase, k=8)), "email": f"u{i}@x.com"}
        for i in range(2000)
    ]
    pairs = candidate_pairs(rows, "email", ["name"])
    all_pairs = 2000 * 1999 // 2
    assert len(pairs) < all_pairs * 0.05


def test_similarity_exact_email_is_one():
    a = {"email": "A@x.com", "name": "a"}
    b = {"email": "a@x.com ", "name": "zzz"}
    assert _similarity(a, b, "email", ["name"]) == 1.0
