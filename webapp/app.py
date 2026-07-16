from flask import Flask, render_template, jsonify

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/overview")
def overview():
    return jsonify({
        "project": "Post-Disaster Recovery Intelligence",
        "events": 7,
        "primary_model": "Transformer",
        "auroc": 0.9264,
        "top30_recall": 0.9324,
        "status": "Research-stage recovery screening system"
    })


if __name__ == "__main__":
    app.run(debug=True)