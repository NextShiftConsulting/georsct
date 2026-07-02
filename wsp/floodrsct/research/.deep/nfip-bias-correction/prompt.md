Research how to statistically correct for known biases in NFIP (National Flood Insurance Program) claims data when using it as a proxy for actual flood damage at the ZCTA (ZIP Code Tabulation Area) level in the United States.

Context: We're building a flood damage prediction framework (FloodRSCT) that uses NFIP claims as one outcome variable. We've found that NFIP claims are systematically corrupted through at least 7 mechanisms: low penetration (~4-5% nationally), Katrina wind-vs-water fraud, Sandy engineering report fraud, post-disaster policy churn, Risk Rating 2.0 structural break, private market invisibility (27-40%), and coverage caps distorting loss ratios.

Sub-questions to investigate:
1. What statistical methods exist to correct for insurance selection bias (non-random who-has-insurance) in damage estimation? Look for Heckman correction, propensity score approaches, or domain-specific methods.
2. How do actuarial science and catastrophe modeling handle known undercounting in claims data? What adjustment factors or credibility weighting schemes exist?
3. Has anyone published NFIP penetration rate estimates at sub-county resolution (ZIP, census tract, block group)? These would enable geographic bias correction.
4. What methods exist for detecting and adjusting for structural breaks in insurance claims time series (Risk Rating 2.0 in Oct 2021, Sandy Claims Review 2015-2017)?
5. Are there published bias correction frameworks for combining multiple biased damage sources (NFIP + FEMA IA + SBA) into a less-biased composite?

Save the report to: C:/Users/marti/github/georsct/wsp/floodrsct/research/.deep/nfip-bias-correction/report.md
