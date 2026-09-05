from flask import Flask, request, Response
import subprocess
import json

app = Flask(__name__)

@app.route('/tune', methods=['POST'])
def tune():
    data = request.json
    model_id = data.get("model_id")
    safe_name = data.get("safe_name")
    
    if not model_id or not safe_name:
        return {"error": "Missing model_id or safe_name"}, 400

    def generate():
        cmd = ["python", "zy_abliterate.py", model_id, safe_name]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        for line in process.stdout:
            # Yield as SSE data format
            yield json.dumps({"type": "status", "msg": line.strip()}) + "\n"
            
        process.wait()
        if process.returncode == 0:
            yield json.dumps({"type": "status", "msg": "Abliteration pipeline completed."}) + "\n"
        else:
            yield json.dumps({"type": "status", "msg": f"FATAL ERROR: Pipeline exited with code {process.returncode}"}) + "\n"

    return Response(generate(), mimetype='application/x-ndjson')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
