What government data sources exist as alternatives or complements to NFIP claims data for measuring flood damage at the ZCTA (ZIP Code Tabulation Area) level in the United States?

Context: We are building a geospatial flood damage prediction framework (FloodRSCT). Our current target variable is NFIP claims, but NFIP has severe biases: ~50% penetration in SFHAs, $250K cap, Sandy engineering fraud, growing private market (27-40%). We need alternative or complementary government data sources that capture flood damage independently of insurance status.

Sub-questions to investigate:
1. FEMA Individual Assistance (IA) via OpenFEMA — volume, spatial resolution, API access, fields, biases
2. SBA Disaster Loans — spatial resolution, format, creditworthiness bias, coverage
3. USGS Short-Term Network (STN) High Water Marks — event count, spatial density, suitability as ZCTA target
4. NOAA Storm Events Database — granularity, reliability, known issues
5. HUD CDBG-DR allocations — granularity, reporting standards
6. Any other federal/state data sources that provide flood damage or impact at sub-county resolution

For each source, assess: spatial resolution, temporal coverage, access method, known biases, and suitability as a ZCTA-level flood damage proxy.

Source hierarchy: Government reports > peer-reviewed papers > court filings > research institutions > investigative journalism. Avoid blog posts and SEO aggregators.

Be skeptical. Distinguish "X happened" from "X is believed to happen." When sources conflict, say so. Never cite a source you haven't read.
