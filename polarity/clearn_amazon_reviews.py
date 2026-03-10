import pandas as pd
import fasttext

# -----------------------------------------------------------
# Load FastText language identification model
# -----------------------------------------------------------
MODEL_PATH = "lid.176.ftz"
model = fasttext.load_model(MODEL_PATH)

def detect_lang(text):
    """Returns (lang_code, probability)."""
    if not isinstance(text, str) or text.strip() == "":
        return "unknown", 0.0
    
    labels, scores = model.predict(text.replace("\n", " "))
    lang = labels[0].replace("__label__", "")
    prob = float(scores[0])
    return lang, prob


def is_english(text, threshold=0.60):
    """Check if text is English with confidence >= threshold."""
    lang, prob = detect_lang(text)
    return lang == "en" 


# -----------------------------------------------------------
# Filter function for any CSV file
# -----------------------------------------------------------
def filter_english_reviews(input_csv, output_csv):
    print(f"\n🌎 Filtering English reviews in: {input_csv}")

    df = pd.read_csv(input_csv, header=None, names=["label", "title", "text"])

    print(f"Total rows before filtering: {len(df)}")

    df["is_english"] = df["text"].apply(lambda x: is_english(str(x)))

    df_clean = df[df["is_english"] == True].drop(columns=["is_english"])

    print(f"Rows kept (English only): {len(df_clean)}")
    print(f"Saving cleaned dataset to: {output_csv}")

    df_clean.to_csv(output_csv, index=False, header=False)
    print("Done!\n")


# -----------------------------------------------------------
# Run cleaning on train + test datasets
# -----------------------------------------------------------
filter_english_reviews("amazon_train.csv", "amazon_train_english.csv")
filter_english_reviews("amazon_test.csv", "amazon_test_english.csv")

print("🎉 All datasets cleaned successfully!")
