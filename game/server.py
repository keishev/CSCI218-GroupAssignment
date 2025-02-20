from flask import Flask, jsonify
from flask_cors import CORS
import subprocess
import os
import signal

app = Flask(__name__)
CORS(app)

process = None
model_server = None

@app.route('/start-camera', methods=['GET'])
def start_camera():
    global process, model_server
    if process is None or process.poll() is not None:
        process = subprocess.Popen(['python', 'model.py'], cwd=os.getcwd())
        return jsonify({'status': 'Camera and Model Server started successfully!'})
    else:
        return jsonify({'status': 'Camera and Model Server are already running.'})

@app.route('/stop-camera', methods=['GET'])
def stop_camera():
    global process, model_server
    if process and process.poll() is None:
        process.terminate()
        process.wait()
    
    if model_server:
        os.killpg(os.getpgid(model_server.pid), signal.SIGTERM)

    return jsonify({'status': 'Camera and Model Server stopped successfully!'})

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5003)
