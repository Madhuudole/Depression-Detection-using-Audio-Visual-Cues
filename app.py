from flask import Flask, render_template, request, jsonify
# replace with your Jupyter script name without the .ipynb extension

app = Flask(__name__)

# Define a route for the homepage
@app.route('/')
def home():
    return render_template('index.html')  # Create an HTML template for your frontend

# Define a route to execute your Jupyter function and return results
@app.route('/run-code', methods=['POST'])
def run_code():
    try:
        result = your_jupyter_script.your_function()  # Call the function in your Jupyter code
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
