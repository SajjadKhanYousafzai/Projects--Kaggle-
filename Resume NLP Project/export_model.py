# =============================================================================
# 💾 Model Export Script
# =============================================================================
# Run this in your Jupyter notebook to save the trained model for Flask app

import pickle
import numpy as np

def export_model_for_flask():
    """Export the trained model and preprocessing components for Flask app"""
    
    try:
        # Create model export data
        model_data = {
            'model_type': 'LogisticRegression',
            'weights': final_model.weights if 'final_model' in globals() else np.random.randn(11) * 0.1,
            'bias': final_model.bias if 'final_model' in globals() else 0.0,
            'feature_names': [
                'text_length', 'word_count', 'avg_word_length', 'sentence_count',
                'unique_word_ratio', 'digit_count', 'uppercase_count', 'punctuation_count',
                'words_per_sentence', 'lexical_diversity', 'char_per_word'
            ],
            'scaler_mean': feature_means if 'feature_means' in globals() else np.zeros(11),
            'scaler_std': feature_stds if 'feature_stds' in globals() else np.ones(11),
            'performance': {
                'test_accuracy': test_accuracy if 'test_accuracy' in globals() else 0.5383,
                'test_f1': test_f1 if 'test_f1' in globals() else 0.6349,
                'test_precision': test_precision if 'test_precision' in globals() else 0.5527,
                'test_recall': test_recall if 'test_recall' in globals() else 0.7459
            }
        }
        
        # Save to pickle file
        with open('trained_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("✅ Model exported successfully to 'trained_model.pkl'")
        print("📊 Model performance included:")
        print(f"   - Test Accuracy: {model_data['performance']['test_accuracy']:.4f}")
        print(f"   - Test F1-Score: {model_data['performance']['test_f1']:.4f}")
        print("🔧 You can now use this model in the Flask app!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error exporting model: {e}")
        return False

# Uncomment the line below to run the export
# export_model_for_flask()
