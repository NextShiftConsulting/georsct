"""PostGIS-backed boundary source adapter.

SQL lives HERE, not in domain code.
Pattern: SQL repo nyc_census_sociodata.sql (tract-keyed schema, GEOID construction).

Post-paper: absorb SQL from rsct-geocert/db/queries/ and db/schema/.
"""

# TODO: Extract from db/scripts/ and db/queries/ after NeurIPS submission.
# Key SQL files to absorb:
#   db/schema/001_scenarios.sql
#   db/schema/006_scenario_pipeline.sql
#   db/queries/f07_summarize_portfolio.sql
