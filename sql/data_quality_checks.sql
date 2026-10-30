-- ============================================================================
-- CRM data-quality checks (SQLite dialect; works with minor tweaks on Postgres / SQL Server)
-- Tables : opportunities_raw, accounts_raw   (loaded straight from the raw CSVs)
-- Param  : :snapshot  -> the extract date, e.g. '2026-10-01'
-- The same 12 checks are implemented in pandas (src/data_quality.py); the pipeline
-- reconciles the two and warns if they ever disagree.
-- ============================================================================
WITH opp AS (
    SELECT
        *,
        CASE WHEN LOWER(stage) LIKE '%won%' OR LOWER(stage) LIKE '%lost%' THEN 1 ELSE 0 END AS is_closed,
        CASE WHEN LOWER(stage) LIKE '%prospect%' OR LOWER(stage) LIKE '%qualif%'
               OR LOWER(stage) LIKE '%proposal%' OR LOWER(stage) LIKE '%negotiat%' THEN 1 ELSE 0 END AS is_open
    FROM opportunities_raw
)
SELECT 'DQ01' AS check_id, COUNT(*) - COUNT(DISTINCT opportunity_id) AS rows_affected FROM opp
UNION ALL
SELECT 'DQ02', COUNT(*) FROM opp
 WHERE stage NOT IN ('Prospecting','Qualification','Proposal','Negotiation','Won','Lost')
UNION ALL
SELECT 'DQ03', COUNT(*) FROM opp WHERE amount IS NULL
UNION ALL
SELECT 'DQ04', COUNT(*) FROM opp WHERE amount <= 0
UNION ALL
SELECT 'DQ05', COUNT(*) FROM opp WHERE is_closed = 1 AND close_date IS NULL
UNION ALL
SELECT 'DQ06', COUNT(*) FROM opp WHERE close_date < created_date
UNION ALL
SELECT 'DQ07', COUNT(*) FROM opp WHERE owner_id IS NULL
UNION ALL
SELECT 'DQ08', COUNT(*) FROM opp
 WHERE account_id NOT IN (SELECT account_id FROM accounts_raw)
UNION ALL
SELECT 'DQ09', COUNT(*) FROM opp WHERE lead_source IS NULL
UNION ALL
SELECT 'DQ10', COUNT(*) FROM opp
 WHERE is_open = 1 AND last_modified_date < date(:snapshot, '-90 day')
UNION ALL
SELECT 'DQ11', COUNT(*) FROM opp
 WHERE is_open = 1 AND close_date < :snapshot
UNION ALL
SELECT 'DQ12', COUNT(*) FROM accounts_raw
 WHERE industry NOT IN ('Mining','Utilities','Transport','Government','Oil & Gas','Industrial')
ORDER BY check_id;
