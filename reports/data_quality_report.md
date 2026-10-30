# CRM Data-Quality Report

Snapshot date: **2026-10-01** | Stale threshold: **90 days**

## Issues found in the raw extract

| ID | Table | Issue | Severity | Rows | % of table | Recommended fix |
|----|-------|-------|----------|-----:|-----------:|-----------------|
| DQ02 | opportunities | Non-standard stage labels | Medium | 241 | 7.02% | Map to the canonical stage list; replace free-text stage with a picklist. |
| DQ11 | opportunities | Open deal whose expected close date has passed (overdue) | High | 123 | 3.58% | Exclude from forecast; reps to update the close date or close out. |
| DQ10 | opportunities | Open deal not updated for >90 days (stale) | High | 116 | 3.38% | Exclude from forecast; reps to re-qualify or close out. |
| DQ09 | opportunities | Missing lead source | Low | 103 | 3.00% | Set to 'Unknown'; make Lead Source mandatory at creation. |
| DQ01 | opportunities | Duplicate opportunity IDs (extra rows) | High | 90 | 2.62% | Keep the most recently modified record per ID; add a unique constraint in the CRM export. |
| DQ05 | opportunities | Closed deal with no close date | High | 75 | 2.19% | Impute from last-modified date (flagged); make Close Date mandatory on closure. |
| DQ07 | opportunities | Deal has no owner | Medium | 52 | 1.52% | Assign to UNASSIGNED and route to sales ops for allocation. |
| DQ03 | opportunities | Missing deal amount | High | 50 | 1.46% | Exclude from revenue maths; make Amount mandatory beyond the Qualification stage. |
| DQ06 | opportunities | Close date earlier than created date | High | 40 | 1.17% | Treat as invalid, repair as above; add a validation rule (Close >= Created). |
| DQ08 | opportunities | Deal linked to an account that does not exist (orphan) | Medium | 33 | 0.96% | Map to an 'Unknown' account; find and restore the missing account records. |
| DQ04 | opportunities | Zero or negative deal amount | High | 27 | 0.79% | Exclude from revenue maths; add a validation rule (Amount > 0). |
| DQ12 | accounts | Non-standard industry labels | Low | 25 | 7.81% | Standardise to the canonical industry list; use a picklist. |

## What the cleaning step did

| Action | Rows |
|--------|-----:|
| Stage labels standardised | 241 |
| Duplicate rows removed | 90 |
| Invalid amounts nulled | 76 |
| Closed deals close date imputed | 107 |
| Open deals close date blanked | 3 |
| Missing owner assigned | 50 |
| Stale open deals flagged | 113 |
| Overdue open deals flagged | 117 |
| Rows out | 3,342 |
| Orphan accounts remapped | 33 |

Every repaired record is flagged with a `dq_*` column in `data/processed/fact_opportunities.csv`, so nothing is changed silently.
