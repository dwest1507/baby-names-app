# Baby Names Explorer

A web application serving 145 years of Social Security Administration (SSA) baby name data with trend charts, ARIMA forecasts, and a Groq-powered natural-language SQL chatbot.

## Language

**Observed Rows**:
Baby name records where at least 5 births were recorded for a given name, sex, and year in SSA records.
_Avoid_: Raw rows, zero-padded rows, fabricated entries

**Built Database**:
The deployable SQLite database artifact containing observed rows, canonical indexes, and precomputed forecast models.
_Avoid_: Source database, raw database, sample database

**Sample Database**:
A lightweight SQLite database with a subset of historical names used for fast local development and CI testing.
_Avoid_: Test database, stub database

**Precomputed Forecast**:
A 5-year statistical forecast (ARIMA/ETS with empirical confidence intervals) generated offline for an eligible baby name.
_Avoid_: Dynamic forecast, live prediction

**Query Resource Budget**:
A wall-clock execution deadline enforced on model-generated SQL statements to prevent worker pool starvation from runaway queries.
_Avoid_: Query timeout, SQL gate
