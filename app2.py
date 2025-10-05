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


@app.route("/model", methods=["GET", "POST"])
def model_page():
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
    return render_template("model.html")  # <-- render the model upload template


@app.route("/predict/<filename>")
def predict(filename):
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(input_path):
        flash("Uploaded file not found.")
        return redirect(url_for("index"))

    try:
        import traceback
        # Step 1: read CSV safely
        df = pd.read_csv(input_path, engine="python", on_bad_lines="skip")
        print("Loaded CSV:", input_path, "shape:", df.shape)
        # Step 2: apply preprocessing if available
        expected_cols = getattr(preprocessor, "feature_names_in_", None)

        if expected_cols is not None:
            df_safe = df.copy()
            for col in expected_cols:
                if col not in df_safe.columns:
                    df_safe[col] = 0
            df_safe = df_safe[expected_cols]
        else:
            df_safe = df.copy()

        # Step 3: transform features
        if preprocessor is not None:
            df_safe.fillna(0, inplace=True)
            X = preprocessor.transform(df_safe)
            print("Transformed X shape:", getattr(X, "shape", None))
        else:
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
        pred_mapping = {0: "CONFIRMED", 1: "CANDIDATE", 2: "FALSE POSITIVE"}
        df.insert(1, "is_exoplanet", [pred_mapping.get(int(p), "Unknown") for p in preds])

        # Safe: only filter by default_flag if the column exists
        if "default_flag" in df.columns:
            df = df[df["default_flag"] == 1]
            df = df.drop(columns=["default_flag"])
        else:
            print("Warning: 'default_flag' column not found in uploaded CSV — skipping that filter.")

        # Step 5: save output CSV
        output_filename = f"predicted_{filename}"
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)
        df.to_csv(output_path, index=False)
        print("Saved predictions to", output_path)

        # Step 6: render HTML preview
        table_html = df.head(200).to_html(classes="table table-striped", index=False, justify="left", escape=False)
        used_model = getattr(model, "__class__", type(model)).__name__

        return render_template(
            "result.html",
            table_html=table_html,
            output_filename=output_filename,
            num_rows=len(df),
            used_model=used_model,
            accuracy=accuracy_score([0]*len(preds), preds) if len(set(preds)) > 1 else "N/A"
        )

    except Exception as e:
        print("ERROR in /predict:", e)
        traceback.print_exc()
        flash(f"Error during preprocessing or prediction: {e}")
        return redirect(url_for("index"))


@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)


@app.route("/exoplanet")
def exoplanet():
    return render_template("exoplanet.html")


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
