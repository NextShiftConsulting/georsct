"""Port: map and visualization rendering.

Contract for producing visual outputs from domain results.
Adapter impls: SVG, DeckGL, Folium, Streamlit.
Pattern: Folium choropleth (AGDS Ch.4-5), Plotly mapbox (Quick-Start Ch.4).
"""

from abc import ABC, abstractmethod

from georsct.flood.domain.readiness_certificate import ReadinessView


class MapRenderer(ABC):
    """Abstract port for map rendering."""

    @abstractmethod
    def render_readiness(
        self,
        view: ReadinessView,
        gdf: "gpd.GeoDataFrame",
        output_path: str,
    ) -> str:
        """Render a readiness view to a visual artifact.

        Returns path to the rendered output.
        """

    @abstractmethod
    def render_choropleth(
        self,
        gdf: "gpd.GeoDataFrame",
        column: str,
        title: str,
        output_path: str,
    ) -> str:
        """Render a choropleth map of a single variable."""

    @abstractmethod
    def render_clusters(
        self,
        gdf: "gpd.GeoDataFrame",
        label_column: str,
        title: str,
        output_path: str,
    ) -> str:
        """Render LISA cluster map (HH/HL/LH/LL quadrants)."""
