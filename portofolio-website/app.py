"""
Flask Application for Data Science Portfolio Website
"""

from flask import Flask, render_template, request, redirect, session, jsonify
import os
import sys
from pathlib import Path

# Tambah path ke src
sys.path.append(str(Path(__file__).parent / "src"))
from predictor import DeliveryTimePredictor

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Dummy users
VALID_USERS = {
    'kinaya': 'datascience123',
    'recruiter': 'hireme2026'
}

# Initialize predictor
predictor = DeliveryTimePredictor()

@app.route('/')
def index():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in VALID_USERS and VALID_USERS[username] == password:
            session['user'] = username
            session['authenticated'] = True
            return redirect('/dashboard')
        else:
            return render_template('login.html', error="Invalid credentials")
    
    return render_template('login.html')

@app.route('/request-access')
def request_access():
    return render_template('request_access.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/dashboard')
def dashboard():
    if not session.get('authenticated'):
        return redirect('/login')
    
    projects = [
        {
            'id': 1,
            'title': 'E-commerce Delivery Time Prediction',
            'description': 'Predict delivery time based on order features',
            'icon': '📦',
            'tags': ['Random Forest', 'Regression', 'Python']
        },
        {
            'id': 2,
            'title': 'Customer Churn Analysis',
            'description': 'Coming soon...',
            'icon': '👥',
            'tags': ['Classification', 'XGBoost', 'SQL']
        },
        {
            'id': 3,
            'title': 'Price Optimization',
            'description': 'Coming soon...',
            'icon': '💰',
            'tags': ['Elasticity', 'Clustering', 'Pandas']
        }
    ]
    
    return render_template('dashboard.html', 
                         user=session.get('user'),
                         projects=projects)

@app.route('/project/<int:project_id>')
def project(project_id):
    if not session.get('authenticated'):
        return redirect('/login')
    
    if project_id == 1:
        return render_template('project1.html')
    else:
        return render_template('coming_soon.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not session.get('authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        
        required_fields = ['total_price', 'n_items', 'total_freight', 'purchase_month']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        order_data = {
            'total_price': float(data['total_price']),
            'n_items': int(data['n_items']),
            'total_freight': float(data['total_freight']),
            'purchase_month': int(data['purchase_month'])
        }
        
        result = predictor.predict_single(order_data)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'unit': result['unit']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['GET'])
def api_search():
    query = request.args.get('q', '')
    
    # Simulasi search results
    results = [
        {'id': 'ORD-001', 'customer': 'João Silva', 'price': 250.00, 'status': 'delivered'},
        {'id': 'ORD-002', 'customer': 'Maria Santos', 'price': 180.50, 'status': 'shipped'},
        {'id': 'ORD-003', 'customer': 'Pedro Oliveira', 'price': 320.00, 'status': 'pending'}
    ]
    
    if query:
        results = [r for r in results if query.lower() in r['id'].lower() 
                  or query.lower() in r['customer'].lower()]
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)