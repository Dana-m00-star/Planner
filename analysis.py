import pandas as pd

# قراءة بيانات المهام
df = pd.read_csv("tasks.csv")

# حساب إجمالي عدد المهام
total_tasks = len(df)

# حساب عدد المهام المكتملة
completed_tasks = len(df[df["status"] == "مكتملة"])

# حساب نسبة الإنجاز
completion_rate = (completed_tasks / total_tasks) * 100

# عرض النتائج
print("Total Tasks:", total_tasks)
print("Completed Tasks:", completed_tasks)
print(f"Completion Rate: {completion_rate:.2f}%")

# حساب نسبة الإنجاز لكل تصنيف
category_completion = (
    df.groupby("category")["status"]
    .apply(lambda x: (x == "مكتملة").mean() * 100)
)

# عرض النتائج
print("\nCategory Completion Rate:")
print(category_completion.round(2))

# تحديد التصنيف الأعلى إنجازًا
most_completed_category = category_completion.idxmax()

print("\nMost Completed Category:", most_completed_category)

# التحليل الرابع: أكثر يوم يتم فيه إنجاز المهام

# تحويل تاريخ الإنجاز إلى نوع datetime
df["completed_date"] = pd.to_datetime(df["completed_date"])

# اختيار المهام المكتملة فقط
completed_df = df[df["status"] == "مكتملة"].copy()

# استخراج اسم اليوم من تاريخ الإنجاز
completed_df["day"] = completed_df["completed_date"].dt.day_name()

# حساب عدد المهام المكتملة في كل يوم
daily_completion = completed_df["day"].value_counts()

# عرض النتائج
print("\nCompleted Tasks by Day:")
print(daily_completion)

# تحديد اليوم الأكثر إنجازًا
most_productive_day = daily_completion.idxmax()

print("\nMost Productive Day:", most_productive_day)

# التحليل الخامس: تحليل الأولويات حسب التصنيف

# حساب نسبة الأولوية العالية في كل تصنيف
category_priority = (
    df.groupby("category")["priority"]
    .apply(lambda x: (x == "عالية").mean() * 100)
)

# عرض النتائج
print("\nHigh Priority Rate by Category:")
print(category_priority.round(2))

# تحديد التصنيف الذي لديه أعلى نسبة أولوية عالية
highest_priority_category = category_priority.idxmax()

print("\nHighest Priority Category:", highest_priority_category)

# التحليل السادس: أكثر فترة زمنية لإنجاز المهام

# استخراج ساعة الإنجاز
completed_df["hour"] = completed_df["completed_date"].dt.hour

# تحديد الفترة الزمنية
def get_time_period(hour):
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 24:
        return "Evening"
    else:
        return "Late Night"

# إضافة الفترة الزمنية لكل مهمة مكتملة
completed_df["time_period"] = completed_df["hour"].apply(get_time_period)

# حساب عدد المهام المكتملة في كل فترة
time_completion = completed_df["time_period"].value_counts()

# عرض النتائج
print("\nCompleted Tasks by Time Period:")
print(time_completion)

# تحديد الفترة الأكثر إنجازًا
most_productive_time = time_completion.idxmax()

print("\nMost Productive Time:", most_productive_time)

# التحليل السابع: تحليل المهام غير المكتملة

# حساب نسبة المهام غير المكتملة في كل تصنيف
incomplete_rate = (
    df.groupby("category")["status"]
    .apply(lambda x: (x == "غير مكتملة").mean() * 100)
)

# عرض النتائج
print("\nIncomplete Tasks Rate by Category:")
print(incomplete_rate.round(2))

# تحديد التصنيف الأكثر إهمالًا
most_neglected_category = incomplete_rate.idxmax()

print("\nMost Neglected Category:", most_neglected_category)

# التحليل الثامن: العلاقة بين مدة المهمة وحالة الإنجاز

# حساب نسبة الإنجاز لكل مدة متوقعة
duration_completion = (
    df.groupby("estimated_minutes")["status"]
    .apply(lambda x: (x == "مكتملة").mean() * 100)
)

# عرض النتائج
print("\nCompletion Rate by Estimated Duration:")
print(duration_completion.round(2))

# تحديد المدة ذات أعلى نسبة إنجاز
best_duration = duration_completion.idxmax()

print("\nDuration with Highest Completion Rate:", best_duration)

# التحليل التاسع: الالتزام بمواعيد المهام

# تحويل التواريخ
df["due_date"] = pd.to_datetime(df["due_date"])
df["completed_date"] = pd.to_datetime(df["completed_date"])

# اختيار المهام المكتملة فقط
completed_tasks_df = df[df["status"] == "مكتملة"].copy()

# مقارنة تاريخ الإنجاز بالموعد النهائي
completed_tasks_df["on_time"] = (
    completed_tasks_df["completed_date"].dt.normalize()
    <= completed_tasks_df["due_date"]
)

# حساب نسبة المهام المنجزة في موعدها
on_time_rate = completed_tasks_df["on_time"].mean() * 100

# حساب نسبة المهام المتأخرة
late_rate = 100 - on_time_rate

# عرض النتائج
print("\nOn-Time Completion Rate:")
print(f"On Time: {on_time_rate:.2f}%")
print(f"Late: {late_rate:.2f}%")


# مراجعة صحة البيانات والتحليلات

print("\nData Validation:")

# عدد المهام
print("Total Tasks:", len(df))

# عدد المهام المكتملة وغير المكتملة
print("Status Counts:")
print(df["status"].value_counts())

# التأكد من عدم وجود قيم مفقودة
print("\nMissing Values:")
print(df.isnull().sum())

# التأكد من عدم تكرار أرقام المهام
print("\nDuplicate Task IDs:")
print(df["task_id"].duplicated().sum())

# التأكد من أن كل مهمة مكتملة لها تاريخ إنجاز
missing_completion = df[
    (df["status"] == "مكتملة") &
    (df["completed_date"].isna())
]

print("\nCompleted Tasks Without Date:", len(missing_completion))

# التأكد من أن تاريخ الإنجاز ليس قبل تاريخ الإنشاء
created_dates = pd.to_datetime(df["created_date"])

invalid_dates = df[
    df["completed_date"].notna() &
    (df["completed_date"] < created_dates)
]

print("Invalid Completion Dates:", len(invalid_dates))


# ============ الرسوم البيانية ============

import matplotlib.pyplot as plt

# pip install arabic-reshaper python-bidi
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    def ar(s):
        return get_display(arabic_reshaper.reshape(str(s)))
except ImportError:
    def ar(s):
        return str(s)

# خط يدعم العربي (يختار أول خط موجود عندك)
plt.rcParams["font.family"] = ["DejaVu Sans"]

for s in (category_completion, category_priority, incomplete_rate):
    s.index = [ar(i) for i in s.index]

status_counts = df["status"].value_counts()
status_counts.index = [ar(i) for i in status_counts.index]

fig, axs = plt.subplots(3, 3, figsize=(18, 14))
axs = axs.flatten()

category_completion.plot.bar(ax=axs[0], title="Completion Rate by Category")
daily_completion.plot.bar(ax=axs[1], title="Completed Tasks by Day")
category_priority.plot.bar(ax=axs[2], title="High Priority % by Category")
time_completion.plot.bar(ax=axs[3], title="Completed Tasks by Time Period")
incomplete_rate.plot.bar(ax=axs[4], title="Incomplete % by Category")
duration_completion.plot.line(ax=axs[5], marker="o", title="Completion % by Duration")

completed_tasks_df["on_time"].value_counts().rename({True: "On Time", False: "Late"}) \
    .plot.pie(ax=axs[6], autopct="%.1f%%", ylabel="", title="On-Time vs Late")
status_counts.plot.pie(ax=axs[7], autopct="%.1f%%", ylabel="", title="Task Status")

axs[8].axis("off")
plt.tight_layout()
plt.savefig("charts.png")
plt.show()
