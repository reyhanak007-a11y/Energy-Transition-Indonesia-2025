from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import os
from model_energi import EnergyTransitionModel

app = Flask(__name__)
model = EnergyTransitionModel()

# Load model yang sudah disimpan
try:
    model.load_model('models/energy_model.joblib')
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error memuat model: {e}")
    print("Membuat model baru...")
    model.load_historical_data()
    model.create_scenarios()

def create_plot(results, scenario_name, historical_data):
    """Membuat plot hasil simulasi"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Pangsa Energi Terbarukan
        ax1.plot(historical_data['year'], historical_data['renewable_share'], 
                 'bo-', label='Data Historis', linewidth=2)
        ax1.plot(results['year'], results['renewable_share'], 
                 'r-', label='Proyeksi', linewidth=2)
        ax1.axhline(y=23, color='g', linestyle='--', label='Target 23%')
        ax1.set_xlabel('Tahun')
        ax1.set_ylabel('Pangsa Energi Terbarukan (%)')
        ax1.set_title(f'Pangsa Energi Terbarukan - {scenario_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Kapasitas Terpasang
        ax2.plot(historical_data['year'], historical_data['renewable_capacity'], 
                 'bo-', label='Data Historis', linewidth=2)
        ax2.plot(results['year'], results['renewable_capacity'], 
                 'r-', label='Proyeksi', linewidth=2)
        ax2.set_xlabel('Tahun')
        ax2.set_ylabel('Kapasitas Terbarukan (MW)')
        ax2.set_title(f'Kapasitas Energi Terbarukan - {scenario_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Investasi
        ax3.plot(results['year'], results['investment'], 'g-', linewidth=2)
        ax3.set_xlabel('Tahun')
        ax3.set_ylabel('Tingkat Investasi')
        ax3.set_title(f'Tingkat Investasi - {scenario_name}')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Infrastruktur
        ax4.plot(results['year'], results['infrastructure'], 'm-', linewidth=2)
        ax4.set_xlabel('Tahun')
        ax4.set_ylabel('Tingkat Infrastruktur')
        ax4.set_title(f'Perkembangan Infrastruktur - {scenario_name}')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert plot to base64 for HTML
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        plt.close(fig)
        
        return plot_url
    except Exception as e:
        print(f"Error creating plot: {e}")
        return None

def create_comparison_plot(historical_data, all_results, scenarios):
    """Membuat plot perbandingan semua skenario"""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot data historis
        ax.plot(historical_data['year'], historical_data['renewable_share'], 
                'ko-', label='Data Historis', linewidth=2)
        
        # Plot setiap skenario
        for scenario_name in scenarios.keys():
            scenario_data = all_results[all_results['scenario'] == scenario_name]
            if len(scenario_data) > 0:
                scenario_info = scenarios[scenario_name]
                ax.plot(scenario_data['year'], scenario_data['renewable_share'],
                       label=scenario_info['name'], color=scenario_info['color'], linewidth=2)
        
        ax.axhline(y=23, color='red', linestyle='--', label='Target 23%', linewidth=2)
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Pangsa Energi Terbarukan (%)')
        ax.set_title('Perbandingan Semua Skenario Kebijakan')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        plt.close(fig)
        
        return plot_url
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
        return None

def create_asean_plot(asean_comparison):
    """Membuat plot perbandingan ASEAN"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        countries = list(asean_comparison.keys())
        shares = list(asean_comparison.values())
        
        # Warna berbeda untuk Indonesia
        colors = ['#ff6b6b' if country == 'Indonesia' else '#4ecdc4' for country in countries]
        
        bars = ax.bar(countries, shares, color=colors, alpha=0.7)
        
        # Tambahkan nilai di atas bar
        for i, (bar, share) in enumerate(zip(bars, shares)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{share:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.axhline(y=23, color='red', linestyle='--', label='Target Indonesia 23%')
        ax.set_ylabel('Pangsa Energi Terbarukan (%)')
        ax.set_title('Perbandingan Pangsa Energi Terbarukan ASEAN')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        plt.close(fig)
        
        return plot_url
    except Exception as e:
        print(f"Error creating ASEAN plot: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data received'})
            
        scenario_name = data.get('scenario', 'business_as_usual')
        end_year = int(data.get('end_year', 2040))
        
        # Validasi input
        if end_year < 2025 or end_year > 2050:
            return jsonify({'success': False, 'error': 'Tahun akhir harus antara 2025-2050'})
        
        # Initial conditions dari data terakhir
        last_data = model.historical_data.iloc[-1]
        initial_conditions = {
            'renewable_capacity': float(last_data['renewable_capacity']),
            'investment': 2.9,
            'infrastructure': 50.0,
            'total_capacity': float(last_data.get('total_capacity', 95400))
        }
        
        # Run simulation
        results = model.run_simulation(scenario_name, initial_conditions, end_year)
        
        # Create plot
        plot_url = create_plot(results, 
                             model.scenarios[scenario_name]['name'],
                             model.historical_data)
        
        if plot_url is None:
            return jsonify({'success': False, 'error': 'Gagal membuat plot'})
        
        # Calculate key metrics - konversi boolean ke string untuk JSON
        final_share = float(results['renewable_share'].iloc[-1])
        target_2025 = float(results[results['year'] == 2025]['renewable_share'].iloc[0])
        
        response = {
            'success': True,
            'plot_url': f"data:image/png;base64,{plot_url}",
            'metrics': {
                'target_2025': round(target_2025, 2),
                'final_share': round(final_share, 2),
                'final_capacity': int(results['renewable_capacity'].iloc[-1]),
                'target_achieved_2025': "Ya" if target_2025 >= 23 else "Tidak",  # Boolean to string
                'target_achieved_final': "Ya" if final_share >= 23 else "Tidak"   # Boolean to string
            },
            'results': results.where(pd.notnull(results), None).to_dict('records')  # Handle NaN values
        }
        
    except Exception as e:
        print(f"Error in simulate: {e}")
        response = {
            'success': False,
            'error': str(e)
        }
    
    return jsonify(response)

@app.route('/compare', methods=['POST'])
def compare_scenarios():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data received'})
            
        end_year = int(data.get('end_year', 2040))
        
        # Validasi input
        if end_year < 2025 or end_year > 2050:
            return jsonify({'success': False, 'error': 'Tahun akhir harus antara 2025-2050'})
        
        # Initial conditions dari data terakhir
        last_data = model.historical_data.iloc[-1]
        initial_conditions = {
            'renewable_capacity': float(last_data['renewable_capacity']),
            'investment': 2.9,
            'infrastructure': 50.0,
            'total_capacity': float(last_data.get('total_capacity', 95400))
        }
        
        # Run all scenarios
        all_results = model.run_all_scenarios(initial_conditions, end_year)
        
        # Create comparison plot
        plot_url = create_comparison_plot(model.historical_data, all_results, model.scenarios)
        
        if plot_url is None:
            return jsonify({'success': False, 'error': 'Gagal membuat plot perbandingan'})
        
        # Calculate comparison metrics - konversi boolean ke string
        comparison_metrics = {}
        for scenario_name in model.scenarios.keys():
            scenario_data = all_results[all_results['scenario'] == scenario_name]
            if len(scenario_data) > 0:
                share_2025 = float(scenario_data[scenario_data['year'] == 2025]['renewable_share'].iloc[0])
                final_share = float(scenario_data['renewable_share'].iloc[-1])
                
                comparison_metrics[scenario_name] = {
                    'name': model.scenarios[scenario_name]['name'],
                    'share_2025': round(share_2025, 2),
                    'final_share': round(final_share, 2),
                    'target_2025_achieved': "Ya" if share_2025 >= 23 else "Tidak"  # Boolean to string
                }
        
        response = {
            'success': True,
            'plot_url': f"data:image/png;base64,{plot_url}",
            'comparison_metrics': comparison_metrics
        }
        
    except Exception as e:
        print(f"Error in compare: {e}")
        response = {
            'success': False,
            'error': str(e)
        }
    
    return jsonify(response)

@app.route('/asean', methods=['GET'])
def asean_comparison():
    """Endpoint untuk data perbandingan ASEAN"""
    try:
        asean_data = model.get_asean_comparison()
        plot_url = create_asean_plot(asean_data)
        
        if plot_url is None:
            return jsonify({'success': False, 'error': 'Gagal membuat plot ASEAN'})
        
        response = {
            'success': True,
            'plot_url': f"data:image/png;base64,{plot_url}",
            'asean_data': asean_data
        }
        
    except Exception as e:
        print(f"Error in asean: {e}")
        response = {
            'success': False,
            'error': str(e)
        }
    
    return jsonify(response)

@app.route('/data', methods=['GET'])
def get_historical_data():
    """Endpoint untuk data historis"""
    try:
        # Konversi NaN ke None untuk JSON serialization
        historical_data = model.historical_data.where(pd.notnull(model.historical_data), None).to_dict('records')
        
        response = {
            'success': True,
            'historical_data': historical_data
        }
        
    except Exception as e:
        print(f"Error in data: {e}")
        response = {
            'success': False,
            'error': str(e)
        }
    
    return jsonify(response)

if __name__ == '__main__':
    # Pastikan folder ada
    os.makedirs('models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('dataset', exist_ok=True)
    
    print("Server starting on http://localhost:5000")

    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
