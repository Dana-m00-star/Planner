import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ========== 1. قراءة البيانات ==========
df = pd.read_csv("tasks.csv")

# ========== 2. تجهيز الأعمدة (Feature Engineering) ==========
df["created_date"] = pd.to_datetime(df["created_date"])
df["due_date"] = pd.to_datetime(df["due_date"])

# يوم الأسبوع (0 = الاثنين ... 6 = الأحد)
df["weekday"] = df["created_date"].dt.weekday

# ساعة إضافة المهمة
df["hour"] = df["created_date"].dt.hour

# كم يوم باقي على الموعد
df["days_to_due"] = (df["due_date"] - df["created_date"].dt.normalize()).dt.days

# عدد المهام في نفس اليوم
df["tasks_same_day"] = df.groupby(df["created_date"].dt.date)["task_id"].transform("count")

# تحويل الأولوية إلى أرقام
df["priority_num"] = df["priority"].map({"منخفضة": 1, "متوسطة": 2, "عالية": 3})

# تحويل التصنيف إلى أعمدة أرقام (0 أو 1)
category_cols = pd.get_dummies(df["category"], prefix="cat").astype(int)
df = pd.concat([df, category_cols], axis=1)

# الهدف: هل المهمة مكتملة؟ (1 = نعم، 0 = لا)
df["target"] = (df["status"] == "مكتملة").astype(int)

# الأعمدة اللي يتعلم منها الموديل
# ملاحظة: ما نستخدم completed_date ولا completion_timing لأنها تكشف الجواب
features = ["weekday", "hour", "days_to_due", "tasks_same_day",
            "priority_num", "estimated_minutes"] + list(category_cols.columns)

X = df[features]
y = df["target"]

# ========== 3. تقسيم البيانات: 80% تدريب و 20% اختبار ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========== 4. تدريب الموديل ==========
model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# ========== 5. قياس الدقة ==========
accuracy = accuracy_score(y_test, model.predict(X_test))
print("===== نتائج الموديل =====")
print(f"دقة الموديل: {accuracy * 100:.0f}%")

# ========== 6. أهم العوامل المؤثرة ==========
importance = pd.Series(model.feature_importances_, index=features)

# دمج أعمدة التصنيف في عامل واحد
importance["category"] = importance[list(category_cols.columns)].sum()
importance = importance.drop(list(category_cols.columns))

names = {
    "weekday": "اليوم",
    "hour": "ساعة الإضافة",
    "days_to_due": "الأيام الباقية على الموعد",
    "tasks_same_day": "عدد مهام اليوم",
    "priority_num": "الأولوية",
    "estimated_minutes": "المدة المتوقعة",
    "category": "التصنيف",
}
english = {
    "weekday": "Day of week",
    "hour": "Hour added",
    "days_to_due": "Days to deadline",
    "tasks_same_day": "Tasks that day",
    "priority_num": "Priority",
    "estimated_minutes": "Estimated minutes",
    "category": "Category",
}

importance = importance.sort_values(ascending=False)
print("\nأهم العوامل المؤثرة على الإنجاز:")
for i, (f, v) in enumerate(importance.items(), 1):
    print(f"{i}. {names[f]} ({v * 100:.0f}%)")

# رسم أهم العوامل
plt.figure(figsize=(8, 5))
plt.barh([english[f] for f in importance.index][::-1], importance.values[::-1] * 100, color="#2E4A7D")
plt.xlabel("Importance (%)")
plt.title("What affects task completion the most?")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("\n(تم حفظ الرسم في feature_importance.png)")

# ========== 7. توقع مهمة جديدة ==========
def predict_task(weekday, hour, days_to_due, tasks_same_day, priority, minutes, category):
    row = {
        "weekday": weekday,
        "hour": hour,
        "days_to_due": days_to_due,
        "tasks_same_day": tasks_same_day,
        "priority_num": {"منخفضة": 1, "متوسطة": 2, "عالية": 3}[priority],
        "estimated_minutes": minutes,
    }
    for col in category_cols.columns:
        row[col] = 1 if col == "cat_" + category else 0
    new_task = pd.DataFrame([row])[features]
    return model.predict_proba(new_task)[0][1] * 100

print("\n===== توقع مهمة جديدة =====")
print("المهمة: مذاكرة فصل 3")
print("اليوم: الجمعة - الساعة 9 مساءً - أولوية عالية - 90 دقيقة")
p = predict_task(weekday=4, hour=21, days_to_due=2, tasks_same_day=13,
                 priority="عالية", minutes=90, category="دراسة")
print(f"احتمال إنجازها: {p:.0f}%", "⚠️" if p < 50 else "✅")

# ========== 8. التوصيات ==========
print("\n===== توصيات لك =====")
day_names = {0: "الاثنين", 1: "الثلاثاء", 2: "الأربعاء", 3: "الخميس",
             4: "الجمعة", 5: "السبت", 6: "الأحد"}

# أفضل وأسوأ يوم
by_day = df.groupby("weekday")["target"].mean()
print(f"✅ أكثر يوم تنجزين فيه: {day_names[by_day.idxmax()]} ({by_day.max() * 100:.0f}%)")
print(f"⚠️ أقل يوم تنجزين فيه: {day_names[by_day.idxmin()]} ({by_day.min() * 100:.0f}%)")

# أفضل فترة
def period(h):
    if h < 12:
        return "الصباح"
    elif h < 18:
        return "الظهر"
    return "المساء"

by_period = df.groupby(df["hour"].apply(period))["target"].mean()
print(f"💡 حطي مهامك المهمة في فترة {by_period.idxmax()} ({by_period.max() * 100:.0f}% إنجاز)")

# عدد المهام في اليوم
many = df[df["tasks_same_day"] > 12]["target"].mean()
few = df[df["tasks_same_day"] <= 12]["target"].mean()
if many < few:
    print(f"⚠️ لما تزيد مهامك عن 12 باليوم ينزل إنجازك من {few * 100:.0f}% إلى {many * 100:.0f}%")

# التصنيف الأكثر إهمالاً
by_cat = df.groupby("category")["target"].mean()
print(f"📌 أكثر تصنيف تأجلينه: {by_cat.idxmin()} ({by_cat.min() * 100:.0f}% إنجاز بس)")

# المهام الطويلة
long_rate = df[df["estimated_minutes"] >= 90]["target"].mean()
short_rate = df[df["estimated_minutes"] < 90]["target"].mean()
if long_rate < short_rate:
    print(f"✂️ قسّمي المهام الطويلة: اللي مدتها ساعة ونص أو أكثر تنجزين منها {long_rate * 100:.0f}% بس")

plt.show()
