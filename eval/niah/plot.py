import json
import glob
import re
import pandas as pd
import seaborn as sns
from metrics import qa_f1_zh_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

tag = 'Meta-Llama-3-8B-Instruct_zh'
tag = 'sft-llama3-qifu-8192-50k-1e5_zh'
tag = 'Qwen1_5-7B-chat_zh'
folder_path = f'../../niah_results/{tag}'  # Replace with your folder path

json_files = glob.glob(f"{folder_path}/*.json")

def zh_score(pred, ref):
    ans_pattern_lower = '刘秀'
    re_res = re.findall(ans_pattern_lower, pred.lower())
    if re_res:
        return 100
    else:
        ref = "王莽在刘秀的手下工作。"
        return qa_f1_zh_score(pred, ref) * 100


score_fn = zh_score


data = []
for file in json_files:
    with open(file, 'r',encoding='utf-8') as f:
        json_data = json.load(f)
        document_depth = json_data.get("depth_percent", None)
        context_length = json_data.get("context_length", None)
        score = json_data.get("score", None)
        score *= 100
        if score_fn is not None:
            if score_fn == qa_f1_zh_score:
                score = score_fn(json_data.get('model_response', ''), "王莽在刘秀的手下工作。") * 100
            elif score_fn:
                score = score_fn(json_data.get('model_response', ''), json_data.get('needle', ''))
        data.append({
            "Document Depth": document_depth,
            "Context Length": context_length,
            "Score": score
        })

df = pd.DataFrame(data)
print(df)

pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
# pivot_table.iloc[:5, :5]

avg_score = pivot_table.values.mean()

# Create a custom colormap. Go to https://coolors.co/ and pick cool colors
cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

# Create the heatmap with better aesthetics
plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
sns.heatmap(
    pivot_table,
    # annot=True,
    fmt="g",
    cmap=cmap,
    vmin=0,
    vmax=100,
    cbar_kws={'label': 'Score'}
)

# More aesthetics
title_lang = 'Chinese'
plt.title(f'{tag}\nNeedle In A HayStack')  # Adds a title
plt.xlabel('Token Limit')  # X-axis label
plt.ylabel('Depth Percent')  # Y-axis label
plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
plt.tight_layout()  # Fits everything neatly into the figure area

# Show the plot
save_folder = 'fig_zh'
plt.show()
# if score_fn is not None:
#     plt.savefig(f'{save_folder}/{result_name}.{score_fn.__name__}.png', bbox_inches='tight')
# else:
#     plt.savefig(f'{save_folder}/{result_name}.f1.png', bbox_inches='tight')