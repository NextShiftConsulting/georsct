"""Use case: Compute spatial probe (weights -> autocorrelation -> regression).

Full spatial analysis pipeline for a scenario.
Chains: W-matrix construction -> Moran's I -> GWR/MGWR -> LISA clusters.
"""

# TODO: Implement post-paper by composing:
#   georsct.domain.kappa.build_weights_*()
#   georsct.domain.turbulence.score_turbulence()
#   georsct.domain.quality.fit_gwr() / fit_mgwr()
#   georsct.domain.turbulence.compute_lisa_clusters()
