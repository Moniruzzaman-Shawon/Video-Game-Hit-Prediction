import gradio as gr
import pandas as pd
import pickle

# Load the Model (Pipeline)
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Dropdown options 
PLATFORMS = [
    "Wii","NES","GB","DS","X360","PS3","PS2","SNES","GBA","PS4","3DS","N64",
    "PS","XB","PC","PSP","2600","GC","WiiU","XOne","PSV","SAT","GEN","DC",
    "SCD","NG","WS","TG16","3DO","GG","PCFX"
]

GENRES = [
    "Action","Sports","Shooter","Role-Playing","Platform","Misc","Racing","Fighting",
    "Simulation","Puzzle","Adventure","Strategy"
]

# Prediction Logic Function
def predict_hit(platform, year, genre, publisher,
                na_sales, eu_sales, jp_sales, other_sales):

    # dataframe with same column names used in training
    input_df = pd.DataFrame([[
        platform, year, genre, publisher,
        na_sales, eu_sales, jp_sales, other_sales
    ]], columns=[
        "Platform", "Year", "Genre", "Publisher",
        "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"
    ])

    pred = model.predict(input_df)[0]

    # Probability 
    proba_text = "N/A"
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0][1]
        proba_text = f"{proba:.4f}"

    # Final label
    result = "HIT (>= 1M Global Sales)" if pred == 1 else "NOT HIT (< 1M Global Sales)"
    return result, proba_text


# Inputs 
inputs = [
    gr.Dropdown(PLATFORMS, label="Platform", value="PS2"),
    gr.Number(label="Year", value=2008),
    gr.Dropdown(GENRES, label="Genre", value="Action"),
    gr.Textbox(label="Publisher", value="Nintendo"),
    gr.Number(label="NA_Sales (millions)", value=0.5),
    gr.Number(label="EU_Sales (millions)", value=0.2),
    gr.Number(label="JP_Sales (millions)", value=0.0),
    gr.Number(label="Other_Sales (millions)", value=0.05),
]


app = gr.Interface(
    fn=predict_hit,
    inputs=inputs,
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Hit Probability")
    ],
    title="Video Game Hit Prediction",
    description="Predict whether a game is HIT or NOT HIT (Global Sales >= 1 million) using a trained Random Forest pipeline."
)

# Launch
app.launch(share=True)
