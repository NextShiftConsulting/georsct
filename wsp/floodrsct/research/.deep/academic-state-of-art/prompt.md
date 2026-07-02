Research question: What is the academic state of the art for flood damage estimation, prediction, and ground truth construction — and does anything like RSCT's certificate-based approach exist?

Context: We are building FloodRSCT, a framework that uses RSCT (Relevance-Stability-Novelty) certificates to evaluate and govern flood damage predictions. Our key findings so far: NFIP R=0.000 for New Orleans (data corruption signal), NYC NFIP degrades at temporal representation level due to Sandy SCRP structural break, Houston shows anti-correlation between NFIP claims and Hazus damage estimates. We want to understand where we sit in the literature.

Sub-questions to investigate:
1. Merz et al. taxonomy of flood damage models — what categories exist, which are dominant
2. Physics-based vs empirical vs ML approaches — Wing et al., JBA, Hazus, depth-damage curves and their known failures
3. ML/AI flood damage prediction — FloodGenome, FloodDamageCast, any deep learning approaches, what target variables they use
4. The "ground truth problem" in flood damage — what fraction of losses are uninsured, how do researchers handle this
5. Multi-source data fusion for flood damage — has anyone combined NFIP + FEMA IA + remote sensing + social media?
6. Does anything like RSCT's certificate framework exist for evaluating flood damage data quality? Any approach that treats R=0.000 as a data validity signal rather than model failure?
7. Key gaps in the literature that our approach addresses

For each paper/approach found, note: authors, year, venue, methodology, target variable, spatial resolution, and key finding. Distinguish VERIFIED (full text read during this session) from ABSTRACT-ONLY (only saw snippet/abstract) from TRAINING-RECALL (know from pre-training knowledge).

Source hierarchy: Peer-reviewed papers > government reports > research institutions. Search by author name + year when known. Avoid blog posts and SEO aggregators.

Be skeptical. If sources conflict, say so and explain which is more credible. Never cite a source you haven't read — if you only saw an abstract, say so.

The files on disk ARE the deliverable. Your final message should just confirm file paths written.
