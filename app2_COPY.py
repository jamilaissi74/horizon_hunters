from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import pandas as pd
import joblib
from werkzeug.utils import secure_filename
from sklearn.metrics import accuracy_score
# Config
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {"csv"}
MODEL_PATH = "gb_comprehensive.joblib"            # your trained model
PREPROCESSOR_PATH = "comprehensive_preprocessor.joblib"  # your preprocessing pipeline

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.secret_key = "change-me-to-something-secret"

# Load model and preprocessor
try:
    model = joblib.load(MODEL_PATH)
    print("Loaded model from", MODEL_PATH)
except Exception as e:
    model = None
    print("No model loaded:", e)

try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("Loaded preprocessor columns:", getattr(preprocessor, "feature_names_in_", None))
except Exception as e:
    preprocessor = None
    print("No preprocessor loaded:", e)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            return redirect(url_for("predict", filename=filename))
        else:
            flash("Please upload a CSV file (.csv)")
            return redirect(request.url)
    return render_template("index.html")


@app.route("/predict/<filename>")
def predict(filename):
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(input_path):
        flash("Uploaded file not found.")
        return redirect(url_for("index"))

    try:
        # Step 1: read CSV safely
        df = pd.read_csv(input_path, engine="python", on_bad_lines="skip")
        print("#######################################################################################################################################",len(df.columns))
        # Step 2: apply preprocessing if available
        expected_cols = getattr(preprocessor, "feature_names_in_", None)

        if expected_cols is not None:
            df_safe = df.copy()
            # Add missing columns
            for col in expected_cols:
                if col not in df_safe.columns:
                    df_safe[col] = 0  # default value
            # Keep only expected columns
            df_safe = df_safe[expected_cols]
        else:
            df_safe = df.copy()

        # Step 3: transform features
        if preprocessor is not None:
            # After reading and aligning the CSV
            df_safe.fillna(0, inplace=True)  # for numeric columns, or use "" for categorical

            X = preprocessor.transform(df_safe)
            print(X)
        else:
            # fallback: use numeric columns only
            numeric_df = df_safe.select_dtypes(include=["number"])
            if numeric_df.shape[1] == 0:
                flash("No numeric columns found and no preprocessor available. Cannot predict.")
                return redirect(url_for("index"))
            X = numeric_df.fillna(0).values

            

        # Step 4: predict
        if model is None:
            flash("No model loaded. Cannot predict.")
            return redirect(url_for("index"))

        preds = model.predict(X)
        #df.insert(1, "is_exoplanet", [int(p) for p in preds])
        # Map numeric predictions to human-readable labels
        pred_mapping = {
            0: "CONFIRMED",
            1: "CANDIDATE",
            2: "FALSE POSITIVE"
        }
        df.insert(1, "is_exoplanet", [pred_mapping.get(int(p), "Unknown") for p in preds])





        # Step 5: save output CSV
        output_filename = f"predicted_{filename}"
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)
        df.to_csv(output_path, index=False)

        # Step 6: render HTML preview
        table_html = df.head(200).to_html(classes="table table-striped", index=False, justify="left", escape=False)
        used_model = getattr(model, "__class__", type(model)).__name__

        # Try to compute a real accuracy if the uploaded CSV contains a true label column.
        # Common label column names to check for.
        possible_label_cols = ["label", "target", "true_label", "is_exoplanet", "y", "ground_truth"]
        found_label_col = None
        for col in possible_label_cols:
            if col in df.columns:
                found_label_col = col
                break

        accuracy_display = None
        pred_distribution = None

        if found_label_col is not None:
            # Extract true labels and make sure lengths match predictions
            true_vals = df[found_label_col].fillna("")
            # If true labels are strings like 'CONFIRMED', map them to numeric if preds are numeric
            try:
                # Attempt to coerce true labels to integers if possible
                true_numeric = pd.to_numeric(true_vals, errors="coerce")
                if true_numeric.isna().any():
                    # Mixed or non-numeric labels: try to map textual labels to the same mapping used above
                    inv_map = {v: k for k, v in pred_mapping.items()}
                    mapped = true_vals.map(lambda x: inv_map.get(str(x).upper(), None))
                    if mapped.isna().all():
                        # Can't map textual labels -> fall back to string comparison with mapped preds
                        preds_str = pd.Series([pred_mapping.get(int(p), "Unknown") for p in preds])
                        accuracy_display = (preds_str.values == true_vals.astype(str).values).mean()
                    else:
                        accuracy_display = accuracy_score(mapped.fillna(-1).astype(int), preds)
                else:
                    accuracy_display = accuracy_score(true_numeric.astype(int), preds)
            except Exception:
                # As a safe fallback, compare stringified values
                preds_str = pd.Series([pred_mapping.get(int(p), "Unknown") for p in preds])
                accuracy_display = (preds_str.values == true_vals.astype(str).values).mean()
        else:
            # No true labels: provide a prediction distribution summary instead of a fake accuracy
            s = pd.Series(preds).map(lambda x: pred_mapping.get(int(x), str(x)))
            counts = s.value_counts()
            percentages = (counts / len(s) * 100).round(2)
            pred_distribution = dict(zip(counts.index.tolist(), [f"{c} ({p}%)" for c, p in zip(counts.tolist(), percentages.tolist())]))
            # Also provide percent predicted as class 0 as a simple indicator
            percent_class0 = (s == pred_mapping.get(0)).mean() * 100
            accuracy_display = {
                "note": "No true labels found in uploaded CSV. Showing prediction distribution instead of accuracy.",
                "distribution": pred_distribution,
                "percent_class_0": round(percent_class0, 2),
            }

        return render_template(
            "result.html",
            table_html=table_html,
            output_filename=output_filename,
            num_rows=len(df),
            used_model=used_model,
            accuracy=accuracy_display,
            pred_distribution=pred_distribution,

        )
    
        

        
    except Exception as e:
        flash(f"Error during preprocessing or prediction: {e}")
        return redirect(url_for("index"))


@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
