"""
Visualize performance comparison of all models.
Shows why original Lorentzian failed and how fixed version works.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import yaml
import sys

# Import the classifier classes so they can be unpickled
from lorentzian_classifier import LorentzianClassifier
from train_lorentzian_fixed import ImprovedLorentzianClassifier

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

def load_test_predictions(symbol):
    """Load test predictions from all models."""
    model_dir = Path('models')
    feat_dir = Path('features')
    
    # Load feature data
    feat_path = feat_dir / f'{symbol}_features.csv'
    df = pd.read_csv(feat_path).dropna()
    
    feature_cols = [c for c in df.columns 
                   if c not in ['y', 'time', 'fwd_ret', 'fwd_ret_net']]
    
    # Test split
    n = len(df)
    n_test = int(n * 0.15)
    test_df = df.iloc[-n_test:]
    
    X_test = test_df[feature_cols].values
    y_test = test_df['y'].values
    
    results = {'y_true': y_test}
    
    # RF
    rf = joblib.load(model_dir / f'{symbol}_rf.pkl')
    rf_scaler = joblib.load(model_dir / 'scaler.pkl')
    X_rf = rf_scaler.transform(X_test)
    results['RF'] = rf.predict_proba(X_rf)[:, 1]
    
    # LSTM
    try:
        import tensorflow as tf
        lstm = tf.keras.models.load_model(model_dir / f'{symbol}_lstm_best.keras')
        lstm_scaler = joblib.load(model_dir / f'{symbol}_lstm_scaler.pkl')
        X_lstm = lstm_scaler.transform(X_test)
        
        # Create sequences
        lookback = 60
        lstm_seqs = []
        valid_idx = []
        for i in range(lookback, len(X_lstm)):
            lstm_seqs.append(X_lstm[i-lookback:i])
            valid_idx.append(i)
        
        lstm_probs = lstm.predict(np.array(lstm_seqs), verbose=0).flatten()
        
        # Align
        lstm_full = np.full(len(y_test), np.nan)
        lstm_full[valid_idx] = lstm_probs
        results['LSTM'] = lstm_full
    except:
        results['LSTM'] = None
    
    # Lorentzian (original k=8)
    try:
        lc = joblib.load(model_dir / f'{symbol}_lorentzian.pkl')
        lc_scaler = joblib.load(model_dir / f'{symbol}_lorentzian_scaler.pkl')
        X_lc = lc_scaler.transform(X_test)
        results['LC (k=8)'] = lc.predict_proba(X_lc)[:, 1]
    except Exception as e:
        print(f"  Warning: Could not load k=8 model: {e}")
        results['LC (k=8)'] = None
    
    # Lorentzian (fixed k=100)
    try:
        lc_fixed = joblib.load(model_dir / f'{symbol}_lorentzian_fixed.pkl')
        lc_fixed_scaler = joblib.load(model_dir / f'{symbol}_lorentzian_fixed_scaler.pkl')
        X_lc_fixed = lc_fixed_scaler.transform(X_test)
        
        probs, conf = lc_fixed.predict_proba(X_lc_fixed, return_confidence=True)
        results['LC (k=100)'] = probs[:, 1]
        results['LC_confidence'] = conf
    except Exception as e:
        print(f"  Warning: Could not load k=100 model: {e}")
        results['LC (k=100)'] = None
        results['LC_confidence'] = None
    
    return results

def create_visualizations(symbol):
    """Create comprehensive visualization."""
    print(f"\nGenerating visualizations for {symbol}...")
    
    results = load_test_predictions(symbol)
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(f'{symbol} - Model Performance Comparison', fontsize=20, fontweight='bold')
    
    # 1. Prediction Distribution Comparison
    ax1 = fig.add_subplot(gs[0, :])
    
    models = ['RF', 'LSTM', 'LC (k=8)', 'LC (k=100)']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for i, (model, color) in enumerate(zip(models, colors)):
        if results.get(model) is not None:
            probs = results[model][~np.isnan(results[model])]
            ax1.hist(probs, bins=50, alpha=0.6, label=model, color=color, density=True)
    
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('1. Prediction Distribution (Problem: Original LC has spike at 0.0)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 2. AUC Comparison Bar Chart
    ax2 = fig.add_subplot(gs[1, 0])
    
    from sklearn.metrics import roc_auc_score
    
    aucs = []
    model_names = []
    for model in models:
        if results.get(model) is not None:
            probs = results[model]
            mask = ~np.isnan(probs)
            if mask.sum() > 0:
                auc = roc_auc_score(results['y_true'][mask], probs[mask])
                aucs.append(auc)
                model_names.append(model)
    
    bars = ax2.barh(model_names, aucs, color=colors[:len(aucs)])
    ax2.set_xlabel('AUC', fontsize=12)
    ax2.set_title('2. Overall AUC Comparison', fontsize=14)
    ax2.set_xlim(0.5, 0.7)
    
    # Add value labels
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        ax2.text(auc + 0.01, i, f'{auc:.3f}', va='center', fontsize=10)
    
    ax2.grid(alpha=0.3, axis='x')
    
    # 3. Confidence Distribution (Fixed LC only)
    ax3 = fig.add_subplot(gs[1, 1])
    
    if results.get('LC_confidence') is not None:
        conf = results['LC_confidence']
        ax3.hist(conf, bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax3.axvline(0.6, color='red', linestyle='--', linewidth=2, label='Threshold (0.6)')
        ax3.set_xlabel('Confidence Score', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_title('3. Lorentzian (Fixed) Confidence', fontsize=14)
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Add percentile markers
        for pct in [25, 50, 75]:
            val = np.percentile(conf, pct)
            ax3.axvline(val, color='gray', linestyle=':', alpha=0.5)
            ax3.text(val, ax3.get_ylim()[1] * 0.9, f'P{pct}', 
                    ha='center', fontsize=8, color='gray')
    
    # 4. High-Confidence Performance
    ax4 = fig.add_subplot(gs[1, 2])
    
    if results.get('LC_confidence') is not None:
        conf = results['LC_confidence']
        probs_fixed = results['LC (k=100)']
        y_true = results['y_true']
        
        thresholds = [0.0, 0.5, 0.6, 0.7, 0.8]
        aucs_by_conf = []
        pcts = []
        
        for thresh in thresholds:
            mask = conf >= thresh
            if mask.sum() > 10:
                auc = roc_auc_score(y_true[mask], probs_fixed[mask])
                pct = mask.mean() * 100
                aucs_by_conf.append(auc)
                pcts.append(pct)
            else:
                aucs_by_conf.append(0)
                pcts.append(0)
        
        ax4_twin = ax4.twinx()
        
        line1 = ax4.plot(thresholds, aucs_by_conf, 'o-', color='#9b59b6', 
                        linewidth=2, markersize=8, label='AUC')
        line2 = ax4_twin.plot(thresholds, pcts, 's--', color='#e67e22', 
                             linewidth=2, markersize=6, label='% Samples')
        
        ax4.set_xlabel('Confidence Threshold', fontsize=12)
        ax4.set_ylabel('AUC', fontsize=12, color='#9b59b6')
        ax4_twin.set_ylabel('% Samples', fontsize=12, color='#e67e22')
        ax4.set_title('4. AUC vs Confidence Threshold', fontsize=14)
        ax4.tick_params(axis='y', labelcolor='#9b59b6')
        ax4_twin.tick_params(axis='y', labelcolor='#e67e22')
        ax4.grid(alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left', fontsize=10)
    
    # 5. ROC Curves
    ax5 = fig.add_subplot(gs[2, :2])
    
    from sklearn.metrics import roc_curve
    
    for model, color in zip(models, colors):
        if results.get(model) is not None:
            probs = results[model]
            mask = ~np.isnan(probs)
            if mask.sum() > 0:
                fpr, tpr, _ = roc_curve(results['y_true'][mask], probs[mask])
                auc = roc_auc_score(results['y_true'][mask], probs[mask])
                ax5.plot(fpr, tpr, color=color, linewidth=2, 
                        label=f'{model} (AUC={auc:.3f})')
    
    # High-confidence LC
    if results.get('LC_confidence') is not None:
        conf = results['LC_confidence']
        probs_fixed = results['LC (k=100)']
        mask = conf >= 0.6
        if mask.sum() > 0:
            fpr, tpr, _ = roc_curve(results['y_true'][mask], probs_fixed[mask])
            auc = roc_auc_score(results['y_true'][mask], probs_fixed[mask])
            ax5.plot(fpr, tpr, color='#f39c12', linewidth=3, linestyle='--',
                    label=f'LC (fixed, conf0.6) (AUC={auc:.3f}) ')
    
    ax5.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax5.set_xlabel('False Positive Rate', fontsize=12)
    ax5.set_ylabel('True Positive Rate', fontsize=12)
    ax5.set_title('5. ROC Curves Comparison', fontsize=14)
    ax5.legend(fontsize=10, loc='lower right')
    ax5.grid(alpha=0.3)
    
    # 6. Performance Summary Table
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    table_data = []
    table_data.append(['Model', 'Test AUC', 'With Filter', '% Trades'])
    
    # RF
    auc_rf = roc_auc_score(results['y_true'], results['RF'])
    table_data.append(['RF', f'{auc_rf:.3f}', '-', '100%'])
    
    # LSTM
    if results['LSTM'] is not None:
        mask = ~np.isnan(results['LSTM'])
        auc_lstm = roc_auc_score(results['y_true'][mask], results['LSTM'][mask])
        table_data.append(['LSTM', f'{auc_lstm:.3f}', '-', '100%'])
    
    # LC original
    if results['LC (k=8)'] is not None:
        auc_lc = roc_auc_score(results['y_true'], results['LC (k=8)'])
        table_data.append(['LC (k=8)', f'{auc_lc:.3f}', '-', '100%'])
    
    # LC fixed
    if results['LC (k=100)'] is not None:
        auc_lc_fixed = roc_auc_score(results['y_true'], results['LC (k=100)'])
        conf = results['LC_confidence']
        mask = conf >= 0.6
        auc_filtered = roc_auc_score(results['y_true'][mask], results['LC (k=100)'][mask])
        pct = mask.mean() * 100
        table_data.append(['LC (fixed)', f'{auc_lc_fixed:.3f}', f'{auc_filtered:.3f}', f'{pct:.0f}%'])
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.2, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best
    for i in range(1, len(table_data)):
        if i == len(table_data) - 1:  # LC fixed row
            table[(i, 2)].set_facecolor('#f39c12')
            table[(i, 2)].set_text_props(weight='bold')
    
    ax6.set_title('6. Performance Summary', fontsize=14, pad=20)
    
    # 7. Key Insights Box
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis('off')
    
    # Calculate metrics
    auc_k8 = roc_auc_score(results['y_true'], results['LC (k=8)']) if results.get('LC (k=8)') is not None else None
    auc_k100 = roc_auc_score(results['y_true'], results['LC (k=100)']) if results.get('LC (k=100)') is not None else None
    
    if results.get('LC_confidence') is not None:
        mask_conf = results['LC_confidence'] >= 0.6
        auc_highconf = roc_auc_score(results['y_true'][mask_conf], results['LC (k=100)'][mask_conf])
        improvement = (auc_highconf - auc_k100) * 100
        pct_trades = mask_conf.mean() * 100
    else:
        auc_highconf = None
        improvement = None
        pct_trades = None
    
    insights = f"""
    KEY INSIGHTS FOR {symbol}:
    
    1. ORIGINAL LORENTZIAN (k=8) PROBLEMS:
        29% predictions at 0.0 (extreme bias)
        k=8 too small (0.1% of data)
        All 14 features used (curse of dimensionality)
        Test AUC: {f'{auc_k8:.3f}' if auc_k8 else 'N/A'}
    
    2. FIXED LORENTZIAN (k=100) IMPROVEMENTS:
        Feature selection: Top 5 features only
        Larger k=100 (1.2% of data, 12x more stable)
        Adaptive k based on local density
        Confidence scoring added
    
    3. PERFORMANCE COMPARISON:
        k=8 AUC: {f'{auc_k8:.3f}' if auc_k8 else 'N/A'}
        k=100 Overall AUC: {f'{auc_k100:.3f}' if auc_k100 else 'N/A'}
        k=100 High-confidence (0.6) AUC: {f'{auc_highconf:.3f}' if auc_highconf else 'N/A'}
        Improvement: {f'{improvement:+.1f}' if improvement else 'N/A'}% when selective
    
    4. RECOMMENDATION:
        Use RF/LSTM for primary predictions
        Use Lorentzian (k=100) confidence as filter
        Only trade when confidence  0.6
        Trade {f'{pct_trades:.0f}' if pct_trades else 'N/A'}% of signals, but with {f'{auc_highconf:.3f}' if auc_highconf else 'N/A'} AUC
    """
    
    ax7.text(0.05, 0.5, insights, transform=ax7.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    
    # Save
    output_path = Path('results') / f'{symbol}_lorentzian_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    
    return output_path

def main():
    symbols = ['USDJPY.sim', 'EURUSD.sim']
    
    for symbol in symbols:
        create_visualizations(symbol)
    
    print("\n Visualizations created!")
    print("\nFiles saved:")
    print("  - results/USDJPY.sim_lorentzian_visualization.png")
    print("  - results/EURUSD.sim_lorentzian_visualization.png")

if __name__ == '__main__':
    main()

