# 8. Generated SQL runs under a resource budget

Date: 2026-09-06

## Status

Accepted

## Context

The chat endpoint runs SQL that a language model wrote. Four guards stood between that
query and the database: it had to begin with `SELECT`, a keyword blocklist refused the
write verbs, the connection was opened read-only, and at most 1000 rows were fetched.

Every one of those guards is about *what the query may touch*. None of them is about
*what the query may cost*, and cost is the exposure that actually matters here, because
the data is public and the connection cannot write.

Three queries pass all four guards and are measured, on a 20,000-row table, as:

| Query | Cost |
| --- | --- |
| `SELECT COUNT(*) FROM names a, names b` | 4.1s, and quadratic — on the real table it does not return |
| `SELECT length(randomblob(900000000))` | allocates 900 MB, in **one row**, where no row cap can reach it |
| an unbounded `WITH RECURSIVE` | does not terminate |

The row cap does not help with any of them: all the work happens before the first row is
handed back. `run_in_threadpool` means each such query also pins one of the pool's
threads, and the memory case can take the container down with it.

This does not need an attacker. A model asked an ambitious question about 145 years of
names can write a self-join by accident.

The `SELECT`-only rule was also costing real answers. It rejected any query beginning
with `WITH`, and CTEs are how the two-stage questions this database invites — rank and
then filter, compare year against year — are naturally written. The schema notes in the
prompt push the model toward exactly those questions.

## Decision

Generated SQL runs on a connection carrying a budget, not just a permission set:
`database.connect_for_generated_sql` attaches a five-second wall-clock deadline via
SQLite's progress handler and caps a single value at one megabyte.

With the budget in place, a leading `WITH` is accepted. A recursive CTE remains the
worst thing the model can write, and the budget is what makes it survivable; the prompt
asks it not to write one.

The row cap is applied by wrapping — `SELECT * FROM (<query>) LIMIT 1000` — rather than
by editing LIMIT out of the query text. Text editing was wrong in both directions: it
rewrote a LIMIT the model had put inside a subquery, answering a different question than
the one asked, and a trailing `-- comment` could both hide the cap and swallow one
appended after it.

## Consequences

An honest question cannot reach five seconds against this data, so the budget is
invisible in normal use; when it does fire, the asker is told the query took too long
rather than shown SQLite's bare `interrupted`.

The blocklist is now redundant four times over and is kept only so that a mistake
produces a clear message instead of a SQLite error. It should never again be treated as
load-bearing: the read-only connection is what makes writes impossible.

The budget is a constant in `database.py`, not configuration. If a legitimate question
ever hits it, that is a signal to index for it, not to raise the number.

The prompt and the validator now have to be changed together — the prompt describes CTEs
as welcome, an automatic row cap, and a time limit. A test asserts they agree.
