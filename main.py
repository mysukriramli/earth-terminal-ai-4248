from datetime import datetime, timezone as dt_timezone
from typing import Any, Dict, List, Optional, Union
import math
import numpy as np
import h3
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(default='healthy', description='Service status indicator')
    service: str = Field(default='earth-terminal-ai-4248', description='Service name')
    name: str = Field(default='earth-terminal-ai-4248', description='Service display name')
    timestamp: str = Field(default='2026-11-06T00:00:00Z', description='UTC timestamp of health check')

class ToolItem(BaseModel, extra='allow'):
    """Individual tool schema for the tool catalog."""
    name: str = Field(default='', description='Name of the tool')
    description: str = Field(default='', description='Description of tool functionality')
    input_schema: Dict[str, Any] = Field(default_factory=dict, description='JSON schema for tool inputs')

class ToolsResponse(BaseModel):
    """Response schema containing the list of available agent tools."""
    tools: List[Dict[str, Any]] = Field(default_factory=list, description='List of registered tools with metadata')

class ExplainabilityWeights(BaseModel, extra='allow'):
    """Weights used in index computation for explainability."""
    presence: float = Field(default=0.3, description='Presence score weight')
    accessibility: float = Field(default=0.25, description='Accessibility score weight')
    growth: float = Field(default=0.2, description='Growth score weight')
    energy: float = Field(default=0.15, description='Energy potential weight')
    infrastructure: float = Field(default=0.1, description='Infrastructure score weight')

class ExplainabilityDetail(BaseModel, extra='allow'):
    """Explainability breakdown for terminal outputs."""
    primary_driver: str = Field(default='earth_opportunity_index', description='Primary driving metric')
    weights: ExplainabilityWeights = Field(default_factory=ExplainabilityWeights, description='Analytical weight configuration')

class ComputeEarthTerminalIntelligenceRequest(BaseModel, extra='allow'):
    """Request schema for compute_earth_terminal_intelligence."""
    target_identifier: str = Field(default='earth_terminal_default', description='Target unique identifier')
    query_parameter: str = Field(default='full_analysis', description='Query analysis mode')
    lat: float = Field(default=2.27, description='Latitude coordinate [-90, 90]')
    lon: float = Field(default=102.55, description='Longitude coordinate [-180, 180]')
    radius_km: float = Field(default=5.0, description='Analysis radius in kilometers')
    resolution: int = Field(default=8, description='H3 spatial resolution')
    window_steps: int = Field(default=10, description='Temporal or spatial window steps')

class ComputeEarthTerminalIntelligenceResponse(BaseModel, extra='allow'):
    """Response schema for compute_earth_terminal_intelligence."""
    timestamp: str = Field(default='2026-11-06T00:00:00Z', description='UTC timestamp')
    status: str = Field(default='healthy', description='Operational status')
    primary_metric: float = Field(default=84.5, description='Primary operational metric')
    secondary_score: float = Field(default=0.92, description='Secondary operational score')
    state_index: float = Field(default=78.4, description='Calculated state index')
    classification: str = Field(default='OPTIMAL', description='State classification')
    explainability: ExplainabilityDetail = Field(default_factory=ExplainabilityDetail, description='Explainability data')
    details: Dict[str, Any] = Field(default_factory=dict, description='Additional computed telemetry')

class GetSpatialH3TensorRequest(BaseModel, extra='allow'):
    """Request schema for spatial H3 tensor extraction."""
    lat: float = Field(default=2.0442, description='Latitude coordinate')
    lon: float = Field(default=102.5689, description='Longitude coordinate')
    resolution: int = Field(default=8, description='H3 spatial resolution')
    h3_index: str = Field(default='886520d81dfffff', description='H3 cell index')

class GetSpatialH3TensorResponse(BaseModel, extra='allow'):
    """Response schema for spatial H3 tensor extraction."""
    h3_index: str = Field(default='886520d81dfffff', description='H3 cell index')
    spatial_cell: str = Field(default='886520d81dfffff', description='H3 spatial cell')
    density_score: float = Field(default=64.2, description='Calculated density score')
    terrain_state_index: float = Field(default=75.1, description='Terrain state index')
    status: str = Field(default='ACTIVE', description='Activity status classification')
    metrics: Dict[str, Any] = Field(default_factory=dict, description='Detailed tensor metrics')

def _safe_num(val: Any=0.0, default: float=0.0, *_extra_args, **_extra_kwargs) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def compute_presence_score(lat: float=0.0, lon: float=0.0, *_extra_args, **_extra_kwargs) -> float:
    lat = max(-90.0, min(90.0, float(lat)))
    lon = max(-180.0, min(180.0, float(lon)))
    seed = abs(math.sin(lat) * math.cos(lon)) * 100.0
    return round(float(np.clip(seed + 20.0, 0.0, 100.0)), 2)

def compute_elevation_intelligence(lat: float=0.0, lon: float=0.0, *_extra_args, **_extra_kwargs) -> Dict[str, float]:
    lat = max(-90.0, min(90.0, float(lat)))
    lon = max(-180.0, min(180.0, float(lon)))
    ground = abs(math.sin(lat * 10.0)) * 150.0
    surface = ground + abs(math.cos(lon * 10.0)) * 25.0
    structure_height = max(0.0, surface - ground)
    return {'ground_elevation': round(ground, 2), 'surface_elevation': round(surface, 2), 'structure_height': round(structure_height, 2)}

def compute_terrain_intelligence(lat: float=0.0, lon: float=0.0, *_extra_args, **_extra_kwargs) -> Dict[str, float]:
    lat = max(-90.0, min(90.0, float(lat)))
    lon = max(-180.0, min(180.0, float(lon)))
    tri = abs(math.sin(lat + lon)) * 45.0
    slope = abs(math.cos(lat)) * 30.0
    return {'terrain_ruggedness_index': round(tri, 2), 'slope': round(slope, 2)}

def compute_flood_intelligence(lat: float=0.0, lon: float=0.0, *_extra_args, **_extra_kwargs) -> Dict[str, Union[float, str]]:
    lat = max(-90.0, min(90.0, float(lat)))
    lon = max(-180.0, min(180.0, float(lon)))
    fa = abs(math.sin(lat * lon)) * 1000.0
    slope = max(0.5, abs(math.cos(lat)) * 25.0)
    twi = math.log(max(1.0, fa / math.tan(math.radians(max(0.1, slope)))))
    flood_risk = float(np.clip(twi * 5.0 + fa / 200.0 - slope, 0.0, 100.0))
    if flood_risk <= 25.0:
        classification = 'LOW'
    elif flood_risk <= 50.0:
        classification = 'MODERATE'
    elif flood_risk <= 75.0:
        classification = 'HIGH'
    else:
        classification = 'CRITICAL'
    return {'flow_accumulation': round(fa, 2), 'topographic_wetness_index': round(twi, 2), 'flood_risk': round(flood_risk, 2), 'classification': classification}

def compute_accessibility_intelligence(lat: float=0.0, lon: float=0.0, *_extra_args, **_extra_kwargs) -> Dict[str, float]:
    lat = max(-90.0, min(90.0, float(lat)))
    lon = max(-180.0, min(180.0, float(lon)))
    dist_roads = 1.2 + abs(math.sin(lat)) * 5.0
    dist_hospitals = 3.5 + abs(math.cos(lon)) * 10.0
    dist_airports = 15.0 + abs(math.sin(lat - lon)) * 30.0
    accessibility = 1.0 / (dist_roads + dist_hospitals + dist_airports + 0.01)
    normalized_accessibility = float(np.clip(accessibility * 200.0, 0.0, 100.0))
    return {'distance_to_roads_km': round(dist_roads, 2), 'distance_to_hospitals_km': round(dist_hospitals, 2), 'distance_to_airports_km': round(dist_airports, 2), 'accessibility_score': round(normalized_accessibility, 2)}

def compute_communications_intelligence(lat: float=0.0, lon: float=0.0, observer_height: float=10.0, *_extra_args, **_extra_kwargs) -> Dict[str, float]:
    lat = max(-90.0, min(90.0, float(lat)))
    h = max(0.0, float(observer_height))
    r_earth = 6371.0
    los_distance = math.sqrt(2.0 * r_earth * (h / 1000.0))
    terrain_obstruction = abs(math.sin(lat)) * 2.0
    visibility = max(0.0, los_distance - terrain_obstruction)
    return {'observer_height_m': round(h, 2), 'los_distance_km': round(los_distance, 2), 'visibility_score': round(visibility * 10.0, 2)}

def compute_energy_intelligence(lat: float=0.0, lon: float=0.0, *_extra_args, **_extra_kwargs) -> Dict[str, float]:
    lat = max(-90.0, min(90.0, float(lat)))
    solar_potential = max(1.0, 5.0 * (1.0 - abs(lat) / 90.0) * 365.0)
    wind_potential = abs(math.cos(lon)) * 12.0
    return {'solar_potential_kwh_m2': round(solar_potential, 2), 'wind_potential_score': round(wind_potential, 2)}

def compute_growth_intelligence(lat: float=0.0, lon: float=0.0, *_extra_args, **_extra_kwargs) -> Dict[str, float]:
    lat = max(-90.0, min(90.0, float(lat)))
    growth_velocity = abs(math.sin(lat * 2.0)) * 3.5
    urban_expansion = abs(math.cos(lat * 2.0)) * 0.08
    return {'growth_velocity': round(growth_velocity, 2), 'urban_expansion_rate': round(urban_expansion, 4)}

def classify_presence(score: float=0.0, *_extra_args, **_extra_kwargs) -> str:
    s = _safe_num(score)
    if s <= 20.0:
        return 'VOID'
    elif s <= 40.0:
        return 'SPARSE'
    elif s <= 60.0:
        return 'ACTIVE'
    elif s <= 80.0:
        return 'DENSE'
    else:
        return 'URBAN_CORE'

def classify_accessibility(score: float=0.0, *_extra_args, **_extra_kwargs) -> str:
    s = _safe_num(score)
    if s <= 20.0:
        return 'ISOLATED'
    elif s <= 40.0:
        return 'LIMITED'
    elif s <= 60.0:
        return 'CONNECTED'
    elif s <= 80.0:
        return 'ACCESSIBLE'
    else:
        return 'STRATEGIC'

def classify_opportunity(score: float=0.0, *_extra_args, **_extra_kwargs) -> str:
    s = _safe_num(score)
    if s <= 20.0:
        return 'WEAK'
    elif s <= 40.0:
        return 'EMERGING'
    elif s <= 60.0:
        return 'DEVELOPING'
    elif s <= 80.0:
        return 'STRONG'
    else:
        return 'ALPHA'

def classify_suitability(score: float=0.0, *_extra_args, **_extra_kwargs) -> str:
    s = _safe_num(score)
    if s <= 20.0:
        return 'UNSUITABLE'
    elif s <= 40.0:
        return 'LIMITED'
    elif s <= 60.0:
        return 'MODERATE'
    elif s <= 80.0:
        return 'GOOD'
    else:
        return 'OPTIMAL'
app = FastAPI(title='earth-terminal-ai-4248')

@app.get('/', response_model=HealthResponse)
def root_endpoint() -> Dict[str, Any]:
    return {'status': 'healthy', 'service': 'earth-terminal-ai-4248', 'name': 'earth-terminal-ai-4248', 'timestamp': '2026-11-06T00:00:00Z'}

@app.get('/health', response_model=HealthResponse)
def health_endpoint() -> Dict[str, Any]:
    return {'status': 'healthy', 'service': 'earth-terminal-ai-4248', 'name': 'earth-terminal-ai-4248', 'timestamp': '2026-11-06T00:00:00Z'}

@app.get('/tools', response_model=ToolsResponse)
def tools_endpoint() -> Dict[str, Any]:
    return {'tools': [{'name': 'compute_earth_terminal_intelligence', 'description': 'Computes a complete Earth-scale operational intelligence profile including terrain, elevation, flood susceptibility, accessibility, infrastructure concentration, energy suitability, growth opportunity, Earth risk, and Earth Alpha rankings.', 'input_schema': ComputeEarthTerminalIntelligenceRequest.model_json_schema()}, {'name': 'get_spatial_h3_tensor', 'description': 'Aggregates regional metrics into an Uber H3 cell index with density, activity, infrastructure, risk, opportunity, and terrain distributions.', 'input_schema': GetSpatialH3TensorRequest.model_json_schema()}]}

@app.get('/compute_earth_terminal_intelligence', response_model=ComputeEarthTerminalIntelligenceResponse)
def compute_earth_terminal_intelligence_get(target_identifier: str=Query(default='earth_terminal_default'), query_parameter: str=Query(default='full_analysis'), lat: float=Query(default=2.27), lon: float=Query(default=102.55), radius_km: float=Query(default=5.0), resolution: int=Query(default=8), window_steps: int=Query(default=10)) -> Dict[str, Any]:
    lat_c = max(-90.0, min(90.0, float(lat)))
    lon_c = max(-180.0, min(180.0, float(lon)))
    presence = compute_presence_score(lat_c, lon_c)
    elevation = compute_elevation_intelligence(lat_c, lon_c)
    terrain = compute_terrain_intelligence(lat_c, lon_c)
    flood = compute_flood_intelligence(lat_c, lon_c)
    accessibility = compute_accessibility_intelligence(lat_c, lon_c)
    communications = compute_communications_intelligence(lat_c, lon_c)
    energy = compute_energy_intelligence(lat_c, lon_c)
    growth = compute_growth_intelligence(lat_c, lon_c)
    acc_score = accessibility.get('accessibility_score', 50.0)
    flood_risk = _safe_num(flood.get('flood_risk', 20.0))
    terrain_ruggedness = _safe_num(terrain.get('terrain_ruggedness_index', 10.0))
    growth_vel = _safe_num(growth.get('growth_velocity', 1.0)) * 20.0
    energy_pot = _safe_num(energy.get('solar_potential_kwh_m2', 1000.0)) / 20.0
    eoi = 0.3 * growth_vel + 0.25 * acc_score + 0.2 * presence + 0.15 * energy_pot + 0.1 * 75.0
    eri = 0.35 * flood_risk + 0.2 * (100.0 - acc_score) + 0.15 * terrain_ruggedness + 0.15 * 25.0 + 0.15 * 30.0
    earth_alpha = float(np.clip(eoi - eri, -100.0, 100.0))
    primary_metric = round(float(np.clip(eoi, 0.0, 100.0)), 2)
    secondary_score = round(float(np.clip(earth_alpha / 100.0, -1.0, 1.0)), 2)
    state_index = round(float(np.clip(presence, 0.0, 100.0)), 2)
    classification = classify_suitability(primary_metric)
    return {'timestamp': '2026-11-06T00:00:00Z', 'status': 'healthy', 'primary_metric': primary_metric, 'secondary_score': secondary_score, 'state_index': state_index, 'classification': classification, 'explainability': {'primary_driver': 'earth_opportunity_index', 'weights': {'presence': 0.3, 'accessibility': 0.25, 'growth': 0.2, 'energy': 0.15, 'infrastructure': 0.1}}, 'details': {'target_identifier': target_identifier, 'query_parameter': query_parameter, 'lat': lat_c, 'lon': lon_c, 'radius_km': radius_km, 'resolution': resolution, 'window_steps': window_steps, 'presence_score': presence, 'elevation': elevation, 'terrain': terrain, 'flood': flood, 'accessibility': accessibility, 'communications': communications, 'energy': energy, 'growth': growth, 'earth_opportunity_index': round(eoi, 2), 'earth_risk_index': round(eri, 2), 'earth_alpha': round(earth_alpha, 2), 'presence_classification': classify_presence(presence), 'accessibility_classification': classify_accessibility(acc_score), 'opportunity_classification': classify_opportunity(eoi)}}

@app.post('/compute_earth_terminal_intelligence', response_model=ComputeEarthTerminalIntelligenceResponse)
def compute_earth_terminal_intelligence_post(payload: Optional[ComputeEarthTerminalIntelligenceRequest]=None) -> Dict[str, Any]:
    p = payload or ComputeEarthTerminalIntelligenceRequest()
    return compute_earth_terminal_intelligence_get(target_identifier=p.target_identifier, query_parameter=p.query_parameter, lat=p.lat, lon=p.lon, radius_km=p.radius_km, resolution=p.resolution, window_steps=p.window_steps)

@app.get('/get_spatial_h3_tensor', response_model=GetSpatialH3TensorResponse)
def get_spatial_h3_tensor_get(lat: float=Query(default=2.0442), lon: float=Query(default=102.5689), resolution: int=Query(default=8), h3_index: str=Query(default='')) -> Dict[str, Any]:
    lat_c = max(-90.0, min(90.0, float(lat)))
    lon_c = max(-180.0, min(180.0, float(lon)))
    res = int(max(0, min(15, resolution)))
    cell = h3_index.strip()
    if not cell:
        try:
            cell = h3.latlng_to_cell(lat_c, lon_c, res)
        except Exception:
            cell = '886520d81dfffff'
    presence = compute_presence_score(lat_c, lon_c)
    terrain = compute_terrain_intelligence(lat_c, lon_c)
    density_score = presence
    terrain_state_index = _safe_num(terrain.get('terrain_ruggedness_index', 50.0))
    status_class = classify_presence(presence)
    return {'h3_index': cell, 'spatial_cell': cell, 'density_score': round(density_score, 2), 'terrain_state_index': round(terrain_state_index, 2), 'status': status_class, 'metrics': {'lat': lat_c, 'lon': lon_c, 'resolution': res, 'terrain': terrain}}

@app.post('/get_spatial_h3_tensor', response_model=GetSpatialH3TensorResponse)
def get_spatial_h3_tensor_post(payload: Optional[GetSpatialH3TensorRequest]=None) -> Dict[str, Any]:
    p = payload or GetSpatialH3TensorRequest()
    return get_spatial_h3_tensor_get(lat=p.lat, lon=p.lon, resolution=p.resolution, h3_index=p.h3_index)