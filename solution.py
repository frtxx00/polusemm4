import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt

def build_features(data):
    agg = (
        data.groupby(["SubjectID", "researchdate", "BrandID", "Brand", "CategoryNameDelivery"])
        .agg(
            count_rows=("QueryText", "size"),
            Weight=("Weight", "first")
        )
        .reset_index()
    )
    agg["daily_ots"] = agg["count_rows"] * agg["Weight"]
    
    brand_day = (
        agg.groupby(["researchdate", "BrandID", "CategoryNameDelivery"])
        .agg(
            brand_day_ots=("daily_ots", "sum"),
            unique_users=("SubjectID", "nunique")
        )
        .reset_index()
    )
    return agg.merge(brand_day, on=["researchdate", "BrandID", "CategoryNameDelivery"])

def add_share(df, alpha=10):
    df["share"] = (df["daily_ots"] + alpha) / (df["brand_day_ots"] + alpha * df["unique_users"])
    return df

def add_deviation(df):
    df["expected_share"] = 1 / df["unique_users"]
    df["deviation"] = df["share"] / (df["expected_share"] + 1e-9)
    return df

def robust_z(x):
    med = np.median(x)
    mad = median_abs_deviation(x)
    return (x - med) / (mad + 1e-9)

def add_event(df):
    hist = (
        df.groupby(["researchdate", "BrandID", "CategoryNameDelivery"])
        .agg(total=("brand_day_ots", "sum"))
        .reset_index()
        .sort_values(["BrandID", "researchdate"])
    )
    hist["baseline"] = hist.groupby("BrandID")["total"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=3).median()
    )
    hist["growth"] = hist["total"] / (hist["baseline"] + 1)
    thr = hist["growth"].quantile(0.995)
    hist["is_event"] = hist["growth"] > thr
    return df.merge(
        hist[["researchdate", "BrandID", "CategoryNameDelivery", "is_event"]],
        on=["researchdate", "BrandID", "CategoryNameDelivery"],
        how="left"
    ).fillna({"is_event": False})

def add_score(df):
    df["deviation_rank"] = df["deviation"].rank(pct=True)
    df["robust_rank"] = df["robust_z"].rank(pct=True)
    df["share_rank"] = df["share"].rank(pct=True)
    df["persistence_rank"] = df["persistence"].rank(pct=True)
    
    df["score"] = (
        0.40 * df["deviation_rank"] +
        0.25 * df["robust_rank"] +
        0.15 * df["share_rank"] +
        0.20 * df["persistence_rank"]
    )
    return df

def plot_before_after_feature(df_before, df_after, col, out_path):
    if col not in df_before.columns:
        return
    b = df_before.groupby(col)["Weight"].sum()
    a = df_after.groupby(col)["Weight"].sum()
    cmp_df = pd.concat([b.rename("before"), a.rename("after")], axis=1).fillna(0)
    cmp_df["pct"] = np.where(cmp_df["before"] > 0, (cmp_df["after"]/cmp_df["before"]-1)*100, 0)
    plt.figure(figsize=(12, 6))
    plt.bar(cmp_df.index.astype(str), cmp_df["pct"])
    plt.xticks(rotation=70, ha="right")
    plt.title(f"{col}: change %")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_querytext_example(df_before, anomaly_pairs, out_path):
    if anomaly_pairs.empty:
        return
    r = anomaly_pairs.iloc[0]
    cols = [c for c in ["SubjectID","researchdate","BrandID","Brand","CategoryNameDelivery","QueryText","Weight"] if c in df_before.columns]
    q = df_before[(df_before["SubjectID"]==r["SubjectID"]) & (df_before["researchdate"]==r["researchdate"])][cols]
    q.to_csv(out_path, index=False)

def plot_brand_before_after(df_before, df_after, brand_id, out_path):
    b = df_before[df_before["BrandID"]==brand_id].groupby("researchdate")["Weight"].sum()
    a = df_after[df_after["BrandID"]==brand_id].groupby("researchdate")["Weight"].sum()
    t = pd.concat([b.rename("before"), a.rename("after")], axis=1).fillna(0)
    plt.figure(figsize=(12, 5))
    plt.plot(t.index, t["before"], label="before")
    plt.plot(t.index, t["after"], label="after")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    input_folder = BASE_DIR / "input_data"
    output_dir = BASE_DIR / "output"
    os.makedirs(output_dir, exist_ok=True)
    created_files = []
    plots_dir = output_dir / "plots"
    os.makedirs(plots_dir, exist_ok=True)
    output_cleaned_path = BASE_DIR / "cleaned_data.csv"
    output_reasons_path = output_dir / "anomaly_reasons.csv"
    output_anomalies_path = output_dir / "anomalies.csv"
    
    if not input_folder.exists() or not input_folder.is_dir():
        print(f"Ошибка: Папка с входными данными не найдена")
        exit(1)
        
    print("Шаг 1. Сбор всех Parquet файлов...")
    parquet_files = [f for f in input_folder.glob("**/*.parquet") if f.is_file()]
    
    data_sheets = []
    for file in parquet_files:
        if file.name.startswith("._") or file.name.startswith("~$"):
            continue
        try:
            df_part = pd.read_parquet(file)
            if not df_part.empty:
                
                if 'Weight' in df_part.columns:
                    df_part['Weight'] = pd.to_numeric(df_part['Weight'], errors='coerce')
                data_sheets.append(df_part)
        except Exception as e:
            print(f"[WARN] Не удалось прочитать {file}: {e}")
            
    df_old = pd.concat(data_sheets, ignore_index=True)
    

        
    df_old["researchdate"] = pd.to_datetime(df_old["researchdate"], errors="coerce")
    df_old["Weight"] = pd.to_numeric(df_old["Weight"], errors="coerce")    
    
    if 'BrandinDelivery' in df_old.columns:
        df_old = df_old[df_old['BrandinDelivery'] == 1].copy() 
        
    df_old = df_old[df_old["CategoryNameDelivery"].notna()].copy()
    df_old = df_old[df_old["researchdate"].notna()].copy()
    df_old = df_old[df_old["Weight"].notna()].copy()

    print(f"Всего загружено строк для анализа: {len(df_old)}")
    
    print("Шаг 2. Расчет признаков и скоринга...")
    agg_df = build_features(df_old)
    agg_df = add_share(agg_df)
    agg_df = add_deviation(agg_df)
    
    agg_df["robust_z"] = agg_df.groupby(
        ["researchdate", "BrandID", "CategoryNameDelivery"]
    )["daily_ots"].transform(robust_z)
    
    agg_df = add_event(agg_df)
    
    user_history = agg_df.groupby("SubjectID").size().rename("activity_days").reset_index()
    agg_df = agg_df.merge(user_history, on="SubjectID", how="left")
    agg_df["persistence"] = agg_df["deviation"] * np.log1p(agg_df["activity_days"])
    
    agg_df = add_score(agg_df)
    
    print("Шаг 3. Определение границ аномалий...")
    base_thr = agg_df["score"].quantile(0.995) 
    agg_df["threshold"] = np.where(agg_df["is_event"], base_thr * 1.05, base_thr)
    agg_df["is_anomaly"] = agg_df["score"] > agg_df["threshold"]
    
    print("Шаг 4. Каскадное удаление респондентов за день по требованию ТЗ...")
    anomaly_pairs = agg_df[agg_df["is_anomaly"]][["SubjectID", "researchdate"]].drop_duplicates()
    anomaly_pairs.to_csv(output_anomalies_path, index=False)
    created_files.append(output_anomalies_path)
    
    agg_df["reason"] = (
    "daily_ots=" + agg_df["daily_ots"].round(3).astype(str) +
    "; share=" + agg_df["share"].round(4).astype(str) +
    "; deviation=" + agg_df["deviation"].round(3).astype(str) +
    "; robust_z=" + agg_df["robust_z"].round(3).astype(str)
)
    
    reasons = agg_df[agg_df["is_anomaly"]][
        [
        "SubjectID", "researchdate", "BrandID", "Brand",
        "CategoryNameDelivery", "daily_ots", "score", "threshold", "reason"
    ]
    ].copy()
    reasons.to_csv(output_reasons_path, index=False)
    created_files.append(output_reasons_path)
    
    # Применяем жесткое каскадное удаление: убираем ВСЕ записи респондента за этот день
    df_cleaned = df_old.merge(
        anomaly_pairs.assign(is_bad_day=True),
        on=['SubjectID', 'researchdate'],
        how='left'
    )
    df_cleaned['is_bad_day'] = (
        df_cleaned['is_bad_day']
        .fillna(False)
        .astype(bool)
    )
    
    # Зануляем вес респондента целиком за этот день
    df_cleaned['Weight'] = np.where(df_cleaned['is_bad_day'] == True, 0.0, df_cleaned['Weight'])
    df_cleaned = df_cleaned.drop(columns=['is_bad_day'])
    
    df_cleaned.to_csv(output_cleaned_path, index=False)
    created_files.append(output_cleaned_path)

    # ========== ПУНКТ 8.1 ГРАФИКИ (PNG)  ==========
    
    before_daily = df_old.groupby("researchdate")["Weight"].sum()
    after_daily = df_cleaned.groupby("researchdate")["Weight"].sum()
    daily_cmp = pd.concat([before_daily.rename("before"), after_daily.rename("after")], axis=1).fillna(0.0)


    #Total OTS before/after
    plt.figure(figsize=(12, 5))
    plt.plot(daily_cmp.index, daily_cmp["before"], label="before")
    plt.plot(daily_cmp.index, daily_cmp["after"], label="after")
    plt.title("Total OTS before/after")
    plt.xlabel("researchdate")
    plt.ylabel("Weight sum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "total_ots_before_after.png", dpi=150)
    plt.close()
    created_files.append(plots_dir / "total_ots_before_after.png")

    #процентное изменение OTS по CategoryDelivery (до \ после)
    cat_col = "CategoryNameDelivery"
    before_cat = df_old.groupby(cat_col)["Weight"].sum()
    after_cat = df_cleaned.groupby(cat_col)["Weight"].sum()
    cat_cmp = pd.concat([before_cat.rename("before"), after_cat.rename("after")], axis=1).fillna(0.0)
    cat_cmp["pct_change"] = np.where(cat_cmp["before"] > 0, (cat_cmp["after"] / cat_cmp["before"] - 1.0) * 100.0, 0.0)
    cat_cmp = cat_cmp.sort_values("pct_change")

    plt.figure(figsize=(12, 6))
    plt.bar(cat_cmp.index.astype(str), cat_cmp["pct_change"])
    plt.xticks(rotation=70, ha="right")
    plt.title("Category OTS change (%)")
    plt.xlabel(cat_col)
    plt.ylabel("Change, %")
    plt.tight_layout()
    plt.savefig(plots_dir / "category_ots_change.png", dpi=150)
    plt.close()
    created_files.append(plots_dir / "category_ots_change.png")

    #сколько уникальных аномальных респондентов (SubjectID) в каждый день (из anomaly_pairs)
    daily_anom = anomaly_pairs.groupby("researchdate")["SubjectID"].nunique().sort_index()
    plt.figure(figsize=(12, 5))
    plt.bar(daily_anom.index.astype(str), daily_anom.values)
    plt.xticks(rotation=70, ha="right")
    plt.title("Daily anomaly count")
    plt.xlabel("researchdate")
    plt.ylabel("Unique anomalous respondents")
    plt.tight_layout()
    plt.savefig(plots_dir / "daily_anomaly_count.png", dpi=150)
    plt.close()
    created_files.append(plots_dir / "daily_anomaly_count.png")
    
    
    # ========== ПУНКТ 8.2 ==========
    #графики «до/после» по характеристикам респондентов: пол, возраст, регион, федеральный округ и т.д.;   
    """    
    for c in ["Пол", "Возраст", "Регион", "Федеральный_округ"]:
        out_file = plots_dir / f"demo_{c}.png"
        plot_before_after_feature(df_old, df_cleaned, c, out_file)

        if out_file.exists():
            created_files.append(out_file)
    """
        
    #графики «до/после» по характеристикам ресурсов: ResourceName, ResourceType, Platform, UseType;
    """
    for c in ["ResourceName", "ResourceType", "Platform", "UseType"]:
        out_file = plots_dir / f"resource_{c}.png"
        plot_before_after_feature(df_old, df_cleaned, c, out_file)

        if out_file.exists():
            created_files.append(out_file)
    """
        
    #графики «до/после» по уровням категорий: CategoryNameDelivery, Category1, Category2, Category3;    
    """
    for c in ["CategoryNameDelivery", "Category1", "Category2", "Category3"]:
        out_file = plots_dir / f"cat_{c}.png"
        plot_before_after_feature(df_old, df_cleaned, c, out_file)
        
        if out_file.exists():
            created_files.append(out_file)
    """
        
    #таблица поисковых запросов QueryText для выбранного аномального респондента и дня;
    """
    save_querytext_example(df_old, anomaly_pairs, output_dir / "querytext_example.csv")
    if not reasons.empty:
        out_file = plots_dir / f"brand_{c}.png"
        plot_before_after_feature(df_old, df_cleaned, c, out_file)
        
        if out_file.exists():
            created_files.append(out_file)
    """
 
    print(" " * 50)
    print("ФИНАЛЬНЫЕ МЕТРИКИ ЭФФЕКТИВНОСТИ")
    old_users = df_old['SubjectID'].nunique()
    new_users = df_cleaned[df_cleaned['Weight'] > 0]['SubjectID'].nunique()
    print(f"1. Доля удаленных респондентов: {(old_users - new_users) / old_users:.4%}")
    
    old_ots = df_old['Weight'].sum()
    new_ots = df_cleaned['Weight'].sum()
    print(f"2. Сохранение OTS (Общая потеря данных): {(old_ots - new_ots) / old_ots:.4%}")
    
    peak_old = df_old.groupby(['researchdate', 'BrandID'])['Weight'].sum().max()
    peak_new = df_cleaned.groupby(['researchdate', 'BrandID'])['Weight'].sum().max()
    print(f"3. Снижение экстремальных пиков: {(peak_old - peak_new) / peak_old:.2%}")
    print(f"   - Максимальный пик ДО: {peak_old:.2f}")
    print(f"   - Максимальный пик ПОСЛЕ: {peak_new:.2f}")
    
    peak_info = (
        df_old.groupby(
            ["researchdate", "Brand"])["Weight"]
        .sum()
        .reset_index()
        .sort_values("Weight", ascending=False)
    )
    
    print(" СОЗДАННЫЕ ФАЙЛЫ  ")
    for p in created_files:
        p = Path(p)
        if p.exists():
            print(f"[OK] {p.name}")

    #print("\nТОП-20 ПИКОВ:")
    #print(peak_info.head(20))
