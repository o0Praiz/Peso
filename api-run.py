"""
Run the Peso API server
"""

from api.api_module import app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
