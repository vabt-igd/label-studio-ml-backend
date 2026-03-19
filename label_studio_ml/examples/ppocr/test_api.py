"""
Tests for PP-OCR ML Backend API.

To run these tests:
    pip install -r requirements-test.txt
    pytest
"""
import pytest
import json
from model import PPOCR


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=PPOCR)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data.get('status') == 'UP'


def test_setup_endpoint(client):
    """Test the setup endpoint."""
    request = {
        'project': '1',
        'schema': '''
        <View>
            <Image name="image" value="$image"/>
            <Polygon name="poly" toName="image"/>
            <TextArea name="transcription" toName="image" perRegion="true"/>
            <Labels name="label" toName="image"><Label value="Text"/></Labels>
        </View>
        ''',
        'hostname': 'http://localhost:8080',
        'access_token': 'test-token'
    }
    response = client.post(
        '/setup',
        data=json.dumps(request),
        content_type='application/json'
    )
    assert response.status_code == 200
