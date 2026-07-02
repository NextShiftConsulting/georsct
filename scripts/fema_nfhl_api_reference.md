# FEMA NFHL API -- Current Endpoints (July 2026)

## 1. Current MapServer URL

The old `/gis/nfhl/` path is dead. The working base URL is:

```
https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer
```

There is also an ArcGIS Online FeatureServer mirror:

```
https://services.arcgis.com/2gdL2gxYNFY2TOUb/arcgis/rest/services/FEMA_National_Flood_Hazard_Layer/FeatureServer
```

## 2. S_FLD_HAZ_AR Layer

**Layer 28** ("Flood Hazard Zones") -- this is S_FLD_HAZ_AR. Key fields: `FLD_ZONE`, `ZONE_SUBTY`, `SFHA_TF`, `STATIC_BFE`, `DFIRM_ID`.

## 3. Query Example (lat/lon point)

```
https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query?geometry=-95.37,29.76&geometryType=esriGeometryPoint&inSR=4326&spatialRel=esriSpatialRelIntersects&outFields=FLD_ZONE,ZONE_SUBTY,SFHA_TF,DFIRM_ID&f=json
```

Replace `-95.37,29.76` with `lon,lat` (note: X,Y order = longitude,latitude).

## 4. OpenFEMA

OpenFEMA (`https://www.fema.gov/api/open`) provides **tabular** NFIP data (claims, policies, disasters). It does **not** support spatial flood zone lookups by lat/lon. Use the NFHL MapServer above for that.
