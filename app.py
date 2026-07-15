"""
Standalone PCA Backend
Flask API for Post-Call Analytics
"""
import os
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'pca-secret-key-change-in-production')
app.config['PCA_INGEST_SECRET'] = os.environ.get('PCA_INGEST_SECRET', 'change-this-secret')

# ClickHouse configuration
app.config['CLICKHOUSE_HOST'] = os.environ.get('CLICKHOUSE_HOST', 'localhost')
app.config['CLICKHOUSE_PORT'] = int(os.environ.get('CLICKHOUSE_PORT', 8123))
app.config['CLICKHOUSE_USER'] = os.environ.get('CLICKHOUSE_USER', 'default')
app.config['CLICKHOUSE_PASSWORD'] = os.environ.get('CLICKHOUSE_PASSWORD', '')
app.config['CLICKHOUSE_DATABASE'] = os.environ.get('CLICKHOUSE_DATABASE', 'voice_analytics')

# AWS configuration
app.config['AWS_REGION'] = os.environ.get('AWS_REGION', 'ap-south-1')
app.config['S3_RECORDINGS_BUCKET'] = os.environ.get('S3_RECORDINGS_BUCKET', 'sahaa-voiceai-recordings')
app.config['PCA_MODEL_ID'] = os.environ.get('PCA_MODEL_ID', 'global.anthropic.claude-haiku-4-5-20251001-v1:0')

# Initialize ClickHouse tables on startup
def init_db():
    try:
        import pca_clickhouse as ch
        ch.ensure_tables()
        print("[PCA] ClickHouse initialized")
    except Exception as e:
        print(f"[PCA] Warning: ClickHouse init failed: {e}")

# Register blueprints
from controllers.pca_controller import pca_bp
from controllers.instore_controller import instore_bp
from controllers.pca_analytics_controller import analytics_bp
app.register_blueprint(pca_bp, url_prefix='/api')
app.register_blueprint(instore_bp, url_prefix='/api')
app.register_blueprint(analytics_bp, url_prefix='/api')

# Initialize on startup
with app.app_context():
    init_db()

@app.route('/health', methods=['GET'])
def health_check():
    return {'status': 'healthy', 'service': 'pca-backend'}, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
