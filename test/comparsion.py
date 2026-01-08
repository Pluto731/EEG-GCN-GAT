import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------
# 1. 配置文件路径
# ---------------------------------------------------------
# 这里的路径对应你提供的四个文件
FILES = {
    "GCN + Pearson": r"E:\Learning\大学课程\计算神经工程\LastWork\checkpoints\gcn_subject_dependent_results.json",
    "GAT + Pearson": r"E:\Learning\大学课程\计算神经工程\LastWork\checkpoints\gat_subject_dependent_results.json",
    "GCN + PLV":     r"E:\Learning\大学课程\计算神经工程\LastWork\checkpoints\gcn_subject_dependent_results_plv.json",
    "GAT + PLV":     r"E:\Learning\大学课程\计算神经工程\LastWork\checkpoints\gat_subject_dependent_results_plv.json"
}

def load_and_parse(name, path):
    """加载 JSON 并提取关键指标"""
    if not os.path.exists(path):
        print(f"⚠️ Warning: File not found: {path}")
        return None
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # 提取 summary
    summary = data['summary']
    
    # 提取所有受试者的平均准确率 (用于画箱线图看稳定性)
    # 假设 json 里有 'avg_acc' 这个 list
    subject_accs = data.get('avg_acc', [])
    
    return {
        "Experiment": name,
        "Overall": summary['overall'],
        "Valence": summary['valence'],
        "Arousal": summary['arousal'],
        "Dominance": summary['dominance'],
        "Liking": summary['liking'],
        "Subject_Accs": subject_accs
    }

def main():
    # 1. 加载数据
    results = []
    for name, path in FILES.items():
        res = load_and_parse(name, path)
        if res:
            results.append(res)
    
    if not results:
        print("No data loaded!")
        return

    df_summary = pd.DataFrame(results)
    
    # ---------------------------------------------------------
    # 2. 打印详细对比表格
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print(f"{'Experiment Strategy':<20} | {'Overall':<8} | {'Valence':<8} | {'Arousal':<8} | {'Dominance':<9} | {'Liking':<8}")
    print("-" * 80)
    for _, row in df_summary.iterrows():
        print(f"{row['Experiment']:<20} | {row['Overall']:.2%}   | {row['Valence']:.2%}   | {row['Arousal']:.2%}   | {row['Dominance']:.2%}     | {row['Liking']:.2%}")
    print("="*80 + "\n")

    # ---------------------------------------------------------
    # 3. 可视化对比
    # ---------------------------------------------------------
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(18, 10))
    
    # 图 1: 总体准确率对比 (Bar Plot)
    ax1 = plt.subplot(2, 2, 1)
    sns.barplot(x="Experiment", y="Overall", data=df_summary, ax=ax1, palette="viridis")
    ax1.set_title("Overall Accuracy Comparison", fontsize=14)
    ax1.set_ylim(0.5, 0.85) # 设置Y轴范围更清晰地看差异
    ax1.bar_label(ax1.containers[0], fmt='%.3f')

    # 图 2: 四大维度详细对比 (Grouped Bar Plot)
    ax2 = plt.subplot(2, 2, 2)
    # 数据变形为长格式
    df_melt = df_summary.melt(id_vars=["Experiment", "Subject_Accs"], 
                              value_vars=["Valence", "Arousal", "Dominance", "Liking"],
                              var_name="Dimension", value_name="Accuracy")
    sns.barplot(x="Dimension", y="Accuracy", hue="Experiment", data=df_melt, ax=ax2, palette="Set2")
    ax2.set_title("Performance by Emotion Dimension", fontsize=14)
    ax2.set_ylim(0.5, 0.9)
    ax2.legend(loc='lower right')

    # 图 3: 模型稳定性对比 (Box Plot - 受试者差异)
    ax3 = plt.subplot(2, 2, 3)
    # 展开 Subject_Accs
    subject_data = []
    for res in results:
        for acc in res['Subject_Accs']:
            subject_data.append({"Experiment": res['Experiment'], "Accuracy": acc})
    df_subjects = pd.DataFrame(subject_data)
    
    sns.boxplot(x="Experiment", y="Accuracy", data=df_subjects, ax=ax3, palette="Pastel1", showmeans=True)
    sns.swarmplot(x="Experiment", y="Accuracy", data=df_subjects, ax=ax3, color=".25", size=3)
    ax3.set_title("Subject Variability (Stability Check)", fontsize=14)
    ax3.set_ylabel("Accuracy per Subject")

    # 图 4: 文本总结区域
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    best_exp = df_summary.loc[df_summary['Overall'].idxmax()]
    worst_exp = df_summary.loc[df_summary['Overall'].idxmin()]
    
    summary_text = (
        f"🏆 Best Performance:\n"
        f"Strategy: {best_exp['Experiment']}\n"
        f"Accuracy: {best_exp['Overall']:.2%}\n\n"
        
        f"📉 Baseline Performance:\n"
        f"Strategy: {worst_exp['Experiment']}\n"
        f"Accuracy: {worst_exp['Overall']:.2%}\n\n"
        
        f"💡 Analysis Highlights:\n"
        f"1. GAT vs GCN: {(df_summary[df_summary['Experiment'].str.contains('GAT')]['Overall'].mean() - df_summary[df_summary['Experiment'].str.contains('GCN')]['Overall'].mean())*100:.2f}% Diff\n"
        f"2. Pearson vs PLV: {(df_summary[df_summary['Experiment'].str.contains('Pearson')]['Overall'].mean() - df_summary[df_summary['Experiment'].str.contains('PLV')]['Overall'].mean())*100:.2f}% Diff"
    )
    ax4.text(0.1, 0.5, summary_text, fontsize=16, va='center', bbox=dict(boxstyle="round", alpha=0.1))

    plt.suptitle("Comparative Analysis of EEG Emotion Recognition Strategies", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(os.path.dirname(list(FILES.values())[0]), 'final_comparison_chart.png')
    plt.savefig(save_path, dpi=300)
    print(f"Chart saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()