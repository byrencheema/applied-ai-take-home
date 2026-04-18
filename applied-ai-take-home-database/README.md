Synthetic Startup Take-Home Database

Files
- synthetic_startup.sqlite: SQLite database file for the assignment

How to use
1. Download and unzip this package.
2. Open synthetic_startup.sqlite in any SQLite-compatible database viewer.
3. Or inspect it from the command line with:
   sqlite3 synthetic_startup.sqlite

Examples
- List tables:
  .tables

- Preview row counts:
  SELECT COUNT(*) FROM scenarios;
  SELECT COUNT(*) FROM artifacts;

- Search long-form evidence:
  SELECT a.title
  FROM artifacts_fts f
  JOIN artifacts a ON a.artifact_id = f.artifact_id
  WHERE artifacts_fts MATCH 'taxonomy rollout'
  LIMIT 10;

Notes
- The main long-form corpus lives in the artifacts table.
- artifacts_fts is the full-text search index for artifact title, summary, and content.
