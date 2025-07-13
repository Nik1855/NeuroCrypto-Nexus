from flask import Flask, jsonify, request
import threading
from core.neural_engine import NeuralEngine

app = Flask(__name__)
neuro_system = None

def initialize_api(engine):
    global neuro_system
    neuro_system = engine
    threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000}).start()

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.json
    symbol = data.get('symbol', 'BTC/USDT')
    prediction = neuro_system.neuromorphic_forward(data['input'])
    return jsonify({'symbol': symbol, 'prediction': prediction.item()})

@app.route('/market-scan', methods=['GET'])
def market_scan():
    symbol = request.args.get('symbol', default='BTC/USDT')
    depth = request.args.get('depth', default=3, type=int)
    scan_results = neuro_system.realtime_sentiment_fusion(symbol)
    return jsonify(scan_results)

@app.route('/system-health', methods=['GET'])
def system_health():
    from modules.self_healing.system_doctor import SystemDoctor
    health_status = SystemDoctor.health_check()
    return jsonify(health_status)