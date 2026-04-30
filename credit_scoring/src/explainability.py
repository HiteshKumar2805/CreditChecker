import shap
import joblib

def explain(model, X_sample):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    print("\nTop contributing features:")
    shap.summary_plot(shap_values, X_sample)