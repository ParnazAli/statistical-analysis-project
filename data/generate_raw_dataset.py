import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

# ---------- گروه‌بندی‌ها ----------
gender = np.random.choice(['Male', 'Female'], n, p=[0.48, 0.52])
education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n, p=[0.25, 0.40, 0.25, 0.10])
city = np.random.choice(['Tehran', 'Isfahan', 'Shiraz', 'Mashhad', 'Tabriz'], n, p=[0.35, 0.2, 0.15, 0.2, 0.1])
group = np.random.choice(['A', 'B', 'C'], n)  # برای ANOVA / کای‌دو
treatment = np.random.choice(['Drug', 'Placebo'], n)  # برای t-test مستقل

# ---------- سن (نرمال) ----------
age = np.random.normal(35, 10, n).clip(18, 75).round(1)

# ---------- درآمد (چوله/غیرنرمال - لاگ‌نرمال) ----------
income = (np.random.lognormal(mean=7.2, sigma=0.5, size=n)).round(0)

# ---------- قد و وزن (همبسته، نرمال) ----------
height = np.random.normal(170, 9, n)
weight = (0.45 * height - 40 + np.random.normal(0, 8, n)).clip(45, 130).round(1)
height = height.round(1)
bmi = (weight / (height/100)**2).round(2)

# ---------- نمرات قبل و بعد (داده زوجی - paired t-test) ----------
score_before = np.random.normal(60, 12, n).clip(0, 100).round(1)
effect = np.where(treatment == 'Drug', np.random.normal(8, 5, n), np.random.normal(1, 5, n))
score_after = (score_before + effect).clip(0, 100).round(1)

# ---------- متغیر وابسته به گروه (برای ANOVA) ----------
group_effect = {'A': 0, 'B': 5, 'C': -3}
performance = np.array([np.random.normal(50 + group_effect[g], 10) for g in group]).round(1)

# ---------- رضایت شغلی (لیکرت 1 تا 5 - ترتیبی) ----------
satisfaction = np.random.choice([1,2,3,4,5], n, p=[0.05,0.10,0.25,0.35,0.25])

# ---------- تعداد فرزندان (شمارشی - پواسون) ----------
children = np.random.poisson(1.5, n)

# ---------- ساعت مطالعه هفتگی (همبسته با نمره) ----------
study_hours = np.random.gamma(shape=2, scale=3, size=n).round(1)
exam_score = (40 + 3*study_hours + np.random.normal(0, 8, n)).clip(0, 100).round(1)

# ---------- متغیر باینری (موفقیت/شکست - رگرسیون لجستیک) ----------
prob_pass = 1 / (1 + np.exp(-(0.08*exam_score - 4)))
passed = np.random.binomial(1, prob_pass)

# ---------- متغیر طبقه‌بندی دوتایی برای کای‌دو ----------
smoker = np.random.choice(['Yes', 'No'], n, p=[0.22, 0.78])
disease = np.where(
    (smoker == 'Yes'),
    np.random.choice(['Yes', 'No'], n, p=[0.35, 0.65]),
    np.random.choice(['Yes', 'No'], n, p=[0.12, 0.88])
)

# ---------- تاریخ ثبت‌نام (سری زمانی) ----------
dates = pd.date_range('2023-01-01', periods=n, freq='4h')
register_date = pd.Series(dates).sample(n, replace=True, random_state=1).reset_index(drop=True)

# ---------- ساخت دیتافریم ----------
df = pd.DataFrame({
    'id': range(1, n+1),
    'gender': gender,
    'age': age,
    'education': education,
    'city': city,
    'group': group,
    'treatment': treatment,
    'income': income,
    'height_cm': height,
    'weight_kg': weight,
    'bmi': bmi,
    'score_before': score_before,
    'score_after': score_after,
    'performance': performance,
    'job_satisfaction_1to5': satisfaction,
    'children_count': children,
    'study_hours_week': study_hours,
    'exam_score': exam_score,
    'passed_exam': passed,
    'smoker': smoker,
    'has_disease': disease,
    'register_date': register_date.dt.strftime('%Y-%m-%d')
})

# ---------- تزریق مقادیر گمشده (برای تمرین missing data) ----------
for col, frac in [('income', 0.03), ('height_cm', 0.02), ('job_satisfaction_1to5', 0.015), ('children_count', 0.01)]:
    idx = np.random.choice(df.index, size=int(frac*n), replace=False)
    df.loc[idx, col] = np.nan

# ---------- تزریق چند داده پرت عمدی (برای تمرین outlier detection) ----------
outlier_idx = np.random.choice(df.index, size=8, replace=False)
df.loc[outlier_idx[:4], 'income'] = df['income'].max() * np.random.uniform(3, 5, 4)
df.loc[outlier_idx[4:], 'exam_score'] = np.random.uniform(-5, 5, 4)

df.to_csv('raw_dataset.csv', index=False, encoding='utf-8-sig')
print(df.shape)
print(df.dtypes)
print(df.isna().sum())
