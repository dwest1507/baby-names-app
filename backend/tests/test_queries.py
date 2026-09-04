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


def test_top_names_excludes_padded_zero_rows():
    # Debra is absent from 2015, so the padding gives her a zero-count row that
    # must never be presented as one of the year's top names.
    names = queries.get_top_names("F", 2015, 10)
    assert names
    assert all(n["total_count"] > 0 for n in names)
    assert "Debra" not in [n["name"] for n in names]


def test_latest_data_year_is_the_newest_year_with_a_recorded_count():
    assert queries.get_latest_data_year() == 2024
