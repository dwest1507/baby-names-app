from app.services import queries


def test_year_range():
    meta = queries.get_year_range()
    assert meta["min_year"] == 1960
    assert meta["max_year"] == 2024


def test_top_names_ordered_by_count():
    names = queries.get_top_names("F", 2015, 5)
    assert names
    counts = [n["total_count"] for n in names]
    assert counts == sorted(counts, reverse=True)
    assert all(n["sex"] == "F" and n["year"] == 2015 for n in names)


def test_name_history_case_insensitive():
    history = queries.get_name_history("emma", "F")
    assert history
    assert history[0]["name"] == "Emma"
    years = [row["year"] for row in history]
    assert years == sorted(years)


def test_name_history_missing_name():
    assert queries.get_name_history("Zzyzx", "M") == []
