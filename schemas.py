from datetime import timezone as dt_timezone
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(default='healthy', description='Service status indicator')
    timestamp: str = Field(default_factory=lambda: datetime.now(dt_timezone.utc).isoformat(), description='UTC timestamp of health check')

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
    radius_km: float = Field(default=5.0, description='Analysis radius in kilometers [0.1, 100]')
    resolution: int = Field(default=8, description='Uber H3 resolution [7, 9]')
    window_steps: int = Field(default=10, description='Temporal or spatial window steps')

class ComputeEarthTerminalIntelligenceResponse(BaseModel, extra='allow'):
    """Response schema for compute_earth_terminal_intelligence."""
    timestamp: str = Field(default_factory=lambda: datetime.now(dt_timezone.utc).isoformat(), description='ISO-8601 UTC timestamp')
    status: str = Field(default='healthy', description='Execution status')
    primary_metric: float = Field(default=84.5, description='Primary computed metric score')
    secondary_score: float = Field(default=0.92, description='Secondary normalized score')
    state_index: float = Field(default=78.4, description='Composite state index value')
    classification: str = Field(default='OPTIMAL', description='Assigned state classification')
    explainability: ExplainabilityDetail = Field(default_factory=ExplainabilityDetail, description='Transparent breakdown of computation drivers')

class GetSpatialH3TensorRequest(BaseModel, extra='allow'):
    """Request schema for get_spatial_h3_tensor."""
    lat: float = Field(default=2.0442, description='Latitude coordinate [-90, 90]')
    lon: float = Field(default=102.5689, description='Longitude coordinate [-180, 180]')
    resolution: int = Field(default=8, description='Uber H3 resolution [7, 9]')

class GetSpatialH3TensorResponse(BaseModel, extra='allow'):
    """Response schema for get_spatial_h3_tensor."""
    h3_index: str = Field(default='886520d81dfffff', description='15-character H3 cell index')
    spatial_cell: str = Field(default='886520d81dfffff', description='Target spatial cell identifier')
    density_score: float = Field(default=64.2, description='Calculated density score')
    terrain_state_index: float = Field(default=75.1, description='Calculated terrain state index')
    status: str = Field(default='ACTIVE', description='Cell activity classification')

class ToolExecutionRequest(BaseModel, extra='allow'):
    """Generic execution request wrapper for tool invocation endpoints."""
    tool_name: str = Field(default='', description='Name of the tool to execute')
    arguments: Dict[str, Any] = Field(default_factory=dict, description='Tool execution arguments')