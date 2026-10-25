Optional folder.
If you place models_full.joblib here (made by colab_export_cell.py), the app uses it for the
"In-sample: trained on all nine" view, provided the scikit-learn version matches.
If the file is missing or the version differs, the app trains the same models itself from
data/training_data.csv, so nothing breaks.
