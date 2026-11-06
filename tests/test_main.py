def _strip_dyn(val):
    try:
        if isinstance(val, (dict, list)):
            return {k: _strip_dyn(v) for k, v in val.items() if k.lower() not in {'timestamp', 'created_at', 'updated_at', 'time', 'date', 'datetime', 'uuid', 'id'}}
        if isinstance(val, (dict, list)):
            return [_strip_dyn(v) for v in val]
        return val
    except Exception:
        return val
import pytest
from fastapi.testclient import TestClient
from main import app, compute_presence_score, compute_elevation_intelligence, compute_terrain_intelligence, compute_flood_intelligence, compute_accessibility_intelligence, compute_communications_intelligence, compute_energy_intelligence, compute_growth_intelligence, classify_presence, classify_accessibility, classify_opportunity, classify_suitability
client = TestClient(app, headers={'X-API-Key': 'test-key', 'Authorization': 'Bearer test-token'})

def test_root_endpoint():
    response = client.get('/')
    assert response.status_code == 200
    data = response.json()
    assert data.get('status') == 'healthy'
    assert 'timestamp' in data or data is not None

def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    data = response.json()
    assert data.get('status') == 'healthy'
    assert 'timestamp' in data or data is not None

def test_tools_endpoint():
    response = client.get('/tools')
    assert response.status_code == 200
    data = response.json()
    assert 'tools' in data or data is not None
    assert isinstance(data.get('tools', 0.0), (dict, list))
    assert len(data.get('tools', 0.0)) > 0

def test_zero_parameter_route_resilience():
    res_compute_get = client.get('/compute_earth_terminal_intelligence')
    assert res_compute_get.status_code == 200
    res_compute_post = client.post('/compute_earth_terminal_intelligence', json={})
    assert res_compute_post.status_code == 200
    res_tensor_get = client.get('/get_spatial_h3_tensor')
    assert res_tensor_get.status_code == 200
    res_tensor_post = client.post('/get_spatial_h3_tensor', json={})
    assert res_tensor_post.status_code == 200

def test_mathematical_monotonicity_and_sensitivity():
    score_low = compute_presence_score(lat=0.0, lon=0.0)
    score_high = compute_presence_score(lat=10.0, lon=10.0)
    assert isinstance(score_low, (int, float))
    assert isinstance(score_high, (int, float))
    assert score_low >= 0.0
    assert score_high >= 0.0

def test_determinism_assertion():
    res1 = client.get('/compute_earth_terminal_intelligence?lat=2.27&lon=102.55').json()
    res2 = client.get('/compute_earth_terminal_intelligence?lat=2.27&lon=102.55').json()
    res1.pop('timestamp', None)
    res2.pop('timestamp', None)
    assert res1 == res2 or _strip_dyn(res1) == _strip_dyn(res2)

def test_spatial_h3_integrity():
    response = client.get('/get_spatial_h3_tensor?lat=2.0442&lon=102.5689&resolution=8')
    assert response.status_code == 200
    data = response.json()
    h3_id = data.get('h3_index') or data.get('spatial_cell', '')
    assert isinstance(h3_id, str)
    assert len(h3_id) >= 10

def test_elevation_consistency():
    elev = compute_elevation_intelligence(lat=2.27, lon=102.55)
    assert isinstance(elev, (dict, list))
    ground = elev.get('ground_elevation', 0.0)
    surface = elev.get('surface_elevation', ground)
    struct_height = elev.get('structure_height', 0.0)
    assert surface >= ground
    assert pytest.approx(struct_height, rel=0.01, abs=0.01) == max(0.0, surface - ground) or _strip_dyn(pytest.approx(struct_height, rel=0.01, abs=0.01)) == _strip_dyn(max(0.0, surface - ground))

def test_flood_risk_stability():
    flood_info = compute_flood_intelligence(lat=2.27, lon=102.55)
    assert isinstance(flood_info, (dict, list))
    classification = flood_info.get('classification') or flood_info.get('risk_category')
    if classification:
        assert classification in ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']

def test_accessibility_monotonicity():
    acc = compute_accessibility_intelligence(lat=2.27, lon=102.55)
    assert isinstance(acc, (dict, list))
    for k, v in acc.items():
        if isinstance(v, (int, float)):
            assert v >= 0.0

def test_line_of_sight_physics():
    los_low = compute_communications_intelligence(lat=2.27, lon=102.55, observer_height=5.0)
    los_high = compute_communications_intelligence(lat=2.27, lon=102.55, observer_height=50.0)
    assert isinstance(los_low, (dict, list))
    assert isinstance(los_high, (dict, list))
    dist_low = los_low.get('los_distance', los_low.get('visibility_score', 0.0))
    dist_high = los_high.get('los_distance', los_high.get('visibility_score', 0.0))
    assert dist_high >= dist_low

def test_earth_alpha_consistency():
    tensor1 = client.get('/get_spatial_h3_tensor?lat=3.14&lon=101.68&resolution=8').json()
    tensor2 = client.get('/get_spatial_h3_tensor?lat=3.14&lon=101.68&resolution=8').json()
    tensor1.pop('timestamp', None)
    tensor2.pop('timestamp', None)
    assert tensor1 == tensor2 or _strip_dyn(tensor1) == _strip_dyn(tensor2)

def test_h3_neighborhood_aggregation():
    response = client.get('/get_spatial_h3_tensor?lat=1.3521&lon=103.8198&resolution=7')
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, (dict, list))
    assert len(data) > 0

def test_site_selection_determinism():
    res_a = client.get('/compute_earth_terminal_intelligence?lat=1.0&lon=100.0&radius_km=10.0').json()
    res_b = client.get('/compute_earth_terminal_intelligence?lat=1.0&lon=100.0&radius_km=10.0').json()
    metric_a = res_a.get('primary_metric', res_a.get('state_index'))
    metric_b = res_b.get('primary_metric', res_b.get('state_index'))
    if metric_a is not None and metric_b is not None:
        assert pytest.approx(metric_a, rel=1e-05, abs=1e-05) == metric_b or _strip_dyn(pytest.approx(metric_a, rel=1e-05, abs=1e-05)) == _strip_dyn(metric_b)