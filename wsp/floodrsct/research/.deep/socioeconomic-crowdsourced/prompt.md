Research socioeconomic indicators, crowdsourced data, and non-traditional data sources that could serve as alternatives or complements to NFIP claims for measuring flood damage/impact at the ZCTA (ZIP Code Tabulation Area) level in the United States.

Context: We're building a flood damage prediction framework (FloodRSCT). We've already researched remote sensing proxies (SAR, NWM, First Street) and government alternatives (FEMA IA, USACE NSI, SBA loans). Now we need the socioeconomic and crowdsourced layer. In our experiments, 311 service request data and USGS high-water marks performed BETTER than NFIP claims as outcome variables in certain cities (Houston, NYC).

Sub-questions to investigate:
1. 311/non-emergency service request data as flood impact proxy -- who has published on this? What cities make 311 data available at ZIP or finer resolution? What request categories correlate with flood damage?
2. Property value changes (Zillow ZTRAX, CoreLogic, county assessor data) as post-flood damage signals -- published methods for isolating flood impact from other price drivers?
3. Social media and crowdsourced flood reports -- geotagged tweets, Waze flood reports, iReport, CoCoRaHS. What's the spatial/temporal resolution? Has anyone validated these against ground truth?
4. Infrastructure disruption data -- power outage maps (EAGLE-I/DOE), cellular network disruption, road closures (Waze, 511 systems), school closures. Available at ZIP level?
5. Displacement and demographic signals -- USPS mail forwarding (change of address), Census population estimates, ACS year-over-year changes, evacuation shelter data. Any published work linking these to flood damage quantification?
6. Health and mortality signals -- CDC WONDER excess mortality, hospital admission surges, EMS call data. Too coarse or actually usable at ZCTA level?

Save the report to: C:/Users/marti/github/georsct/wsp/floodrsct/research/.deep/socioeconomic-crowdsourced/report.md
