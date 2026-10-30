import streamlit as st
import pandas as pd
from datetime import datetime, date
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Planner | سجل المهام",
    page_icon="💙",
    layout="wide"
)

FILE_NAME = "tasks.csv"
TRAINING_FILE = "training_tasks.csv"

CATEGORIES = {
    "study": "دراسة",
    "home": "بيت",
    "exercise": "رياضة",
    "personal": "شخصي"
}

PRIORITIES = {
    "high": "عالية",
    "medium": "متوسطة",
    "low": "منخفضة"
}

COLUMNS = [
    "task_id",
    "task_name",
    "created_date",
    "due_date",
    "estimated_minutes",
    "status",
    "completed_date",
    "category",
    "priority",
    "deleted_date"
]

FEATURES = [
    "weekday",
    "hour",
    "days_to_due",
    "tasks_same_day",
    "priority_num",
    "estimated_minutes",
    "cat_study",
    "cat_home",
    "cat_exercise",
    "cat_personal"
]


# ==============================
# تصميم التطبيق والألوان
# ==============================

st.markdown("""
<style>
.stApp {
    background-color: #192B49;
    color: #F3EBDD;
}

[data-testid="stHeader"] {
    background-color: #192B49;
}

[data-testid="stSidebar"] {
    background-color: #F3EBDD;
}

[data-testid="stSidebar"] * {
    color: #192B49 !important;
}

h1, h2, h3, h4, p, label, .stMarkdown,
[data-testid="stMetricLabel"] {
    color: #F3EBDD;
}

[data-testid="stMetricValue"] {
    color: #F3EBDD;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #F3EBDD;
    border: 1px solid #D9CCB8;
    border-radius: 14px;
    padding: 12px;
}

[data-testid="stVerticalBlockBorderWrapper"] * {
    color: #192B49;
}

.stButton > button,
.stButton > button *,
.stFormSubmitButton > button,
.stFormSubmitButton > button * {
    background-color: #F3EBDD;
    color: #192B49 !important;
    border: 1px solid #D9CCB8;
    border-radius: 10px;
    font-weight: bold;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #A9D6F5;
    color: #192B49 !important;
    border-color: #A9D6F5;
}

.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #F3EBDD;
    color: #192B49;
}

.stTextInput label,
.stNumberInput label,
.stDateInput label,
.stSelectbox label,
.stSlider label {
    color: #F3EBDD !important;
}

hr {
    border-color: #F3EBDD;
}
</style>
""", unsafe_allow_html=True)


# ==============================
# قراءة المهام وحفظها
# ==============================

def load_tasks():

    try:
        df = pd.read_csv(
            FILE_NAME,
            dtype={"task_id": str},
            keep_default_na=False
        )

    except (FileNotFoundError, pd.errors.EmptyDataError):

        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(FILE_NAME, index=False)

    for column in COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df["task_id"] = pd.to_numeric(
        df["task_id"],
        errors="coerce"
    )

    df["estimated_minutes"] = pd.to_numeric(
        df["estimated_minutes"],
        errors="coerce"
    ).fillna(0)

    df["deleted_date"] = df["deleted_date"].fillna("")
    df["completed_date"] = df["completed_date"].fillna("")

    return df[COLUMNS]


def save_tasks(df):
    df.to_csv(FILE_NAME, index=False)


# ==============================
# تجهيز البيانات للتحليل والنموذج
# ==============================

def prepare_analysis_data(data):

    df = data.copy()

    status_map = {
        "complete": 1,
        "completed": 1,
        "مكتملة": 1,
        "منجزة": 1,
        "incomplete": 0,
        "غير مكتملة": 0,
        "قيد التنفيذ": 0
    }

    priority_map = {
        "low": 1,
        "منخفضة": 1,
        "medium": 2,
        "متوسطة": 2,
        "high": 3,
        "عالية": 3
    }

    category_map = {
        "study": "study",
        "دراسة": "study",
        "home": "home",
        "بيت": "home",
        "منزل": "home",
        "exercise": "exercise",
        "رياضة": "exercise",
        "personal": "personal",
        "شخصي": "personal"
    }

    df["target"] = (
        df["status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(status_map)
    )

    df["priority_num"] = (
        df["priority"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(priority_map)
    )

    df["category"] = (
        df["category"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(category_map)
    )

    df["created_date"] = pd.to_datetime(
        df["created_date"],
        errors="coerce"
    )

    df["due_date"] = pd.to_datetime(
        df["due_date"],
        errors="coerce"
    )

    df["completed_date"] = pd.to_datetime(
        df["completed_date"],
        errors="coerce"
    )

    df["estimated_minutes"] = pd.to_numeric(
        df["estimated_minutes"],
        errors="coerce"
    )

    df["weekday"] = df["created_date"].dt.weekday
    df["hour"] = df["created_date"].dt.hour

    df["days_to_due"] = (
        df["due_date"] -
        df["created_date"].dt.normalize()
    ).dt.days

    df["tasks_same_day"] = (
        df.groupby(
            df["created_date"].dt.date
        )["task_id"].transform("count")
    )

    category_columns = [
        "cat_study",
        "cat_home",
        "cat_exercise",
        "cat_personal"
    ]

    for category in [
        "study",
        "home",
        "exercise",
        "personal"
    ]:

        df[f"cat_{category}"] = (
            df["category"] == category
        ).astype(int)

    return df, category_columns


# ==============================
# تحويل مدة المهمة إلى نص
# ==============================

def show_duration(value):

    try:
        total = max(0, int(float(value)))
    except (TypeError, ValueError):
        total = 0

    parts = []

    if total // 1440:
        parts.append(f"{total // 1440} يوم")

    if (total % 1440) // 60:
        parts.append(f"{(total % 1440) // 60} ساعة")

    if total % 60:
        parts.append(f"{total % 60} دقيقة")

    return " و ".join(parts) if parts else "0 دقيقة"


# ==============================
# تحميل بيانات المستخدم
# ==============================

tasks = load_tasks()

active_tasks = tasks[
    tasks["deleted_date"].astype(str).eq("")
].copy()

deleted_tasks = tasks[
    tasks["deleted_date"].astype(str).ne("")
].copy()

completed_count = int(
    (active_tasks["status"] == "complete").sum()
)

incomplete_count = len(active_tasks) - completed_count

completion_rate = (
    completed_count / len(active_tasks) * 100
    if len(active_tasks) else 0
)


# ==============================
# القائمة الرئيسية
# ==============================

st.title("Planner")
st.caption("نظّم مهامك، تابع إنجازك، وخَلّ يومك أسهل")

page = st.sidebar.radio(
    "القائمة الرئيسية",
    [
        "مهامي",
        "إضافة مهمة",
        "سلة المحذوفات",
        "تحليل البيانات",
        "التوقعات والتوصيات"
    ]
)


# ==================================================
# الصفحة الأولى: مهامي
# ==================================================

if page == "مهامي":

    st.header("مهامي")

    c1, c2, c3 = st.columns(3)

    c1.metric("إجمالي المهام", len(active_tasks))
    c2.metric("المهام المكتملة", completed_count)
    c3.metric("المهام المتبقية", incomplete_count)

    st.progress(
        int(completion_rate),
        text=f"نسبة الإنجاز: {completion_rate:.1f}%"
    )

    st.divider()

    search = st.text_input("ابحث عن مهمة")

    filter_status = st.selectbox(
        "حالة المهمة",
        ["الكل", "مكتملة", "غير مكتملة"]
    )

    filter_category = st.selectbox(
        "التصنيف",
        ["الكل"] + list(CATEGORIES.keys()),
        format_func=lambda x:
        "الكل" if x == "الكل" else CATEGORIES[x]
    )

    displayed = active_tasks.copy()

    if search:

        displayed = displayed[
            displayed["task_name"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    if filter_status == "مكتملة":

        displayed = displayed[
            displayed["status"] == "complete"
        ]

    elif filter_status == "غير مكتملة":

        displayed = displayed[
            displayed["status"] == "incomplete"
        ]

    if filter_category != "الكل":

        displayed = displayed[
            displayed["category"] == filter_category
        ]

    if displayed.empty:
        st.info("لا توجد مهام لعرضها.")

    for _, task in displayed.iterrows():

        task_id = int(task["task_id"])

        with st.container(border=True):

            st.subheader(str(task["task_name"]))

            st.write(
                f"التصنيف: "
                f"{CATEGORIES.get(task['category'], task['category'])}"
            )

            st.write(
                f"الأولوية: "
                f"{PRIORITIES.get(task['priority'], task['priority'])}"
            )

            st.write(f"تاريخ الاستحقاق: {task['due_date']}")

            st.write(
                f"المدة المتوقعة: "
                f"{show_duration(task['estimated_minutes'])}"
            )

            is_complete = task["status"] == "complete"

            new_status = st.checkbox(
                "مكتملة",
                value=is_complete,
                key=f"complete_{task_id}"
            )

            if new_status != is_complete:

                idx = tasks.index[
                    tasks["task_id"] == task_id
                ]

                tasks.loc[idx, "status"] = (
                    "complete" if new_status else "incomplete"
                )

                tasks.loc[idx, "completed_date"] = (
                    datetime.now().strftime("%Y-%m-%d %H:%M")
                    if new_status else ""
                )

                save_tasks(tasks)
                st.rerun()

            col1, col2 = st.columns(2)

            if col1.button(
                "حذف المهمة",
                key=f"delete_{task_id}"
            ):

                idx = tasks.index[
                    tasks["task_id"] == task_id
                ]

                tasks.loc[idx, "deleted_date"] = (
                    datetime.now().strftime("%Y-%m-%d %H:%M")
                )

                save_tasks(tasks)
                st.rerun()

            if col2.button(
                "تعديل المهمة",
                key=f"edit_{task_id}"
            ):

                st.session_state["editing_task"] = task_id
                st.rerun()

    # تعديل المهمة

    if "editing_task" in st.session_state:

        task_id = st.session_state["editing_task"]

        selected = tasks[
            tasks["task_id"] == task_id
        ]

        if not selected.empty:

            task = selected.iloc[0]

            st.divider()
            st.subheader("تعديل المهمة")

            parsed_due = pd.to_datetime(
                task["due_date"],
                errors="coerce"
            )

            with st.form("edit_task_form"):

                new_name = st.text_input(
                    "اسم المهمة",
                    value=str(task["task_name"])
                )

                new_category = st.selectbox(
                    "التصنيف",
                    list(CATEGORIES),
                    index=(
                        list(CATEGORIES).index(task["category"])
                        if task["category"] in CATEGORIES else 0
                    ),
                    format_func=lambda x: CATEGORIES[x]
                )

                new_priority = st.selectbox(
                    "الأولوية",
                    list(PRIORITIES),
                    index=(
                        list(PRIORITIES).index(task["priority"])
                        if task["priority"] in PRIORITIES else 0
                    ),
                    format_func=lambda x: PRIORITIES[x]
                )

                new_due = st.date_input(
                    "تاريخ الاستحقاق",
                    value=(
                        parsed_due.date()
                        if pd.notna(parsed_due)
                        else date.today()
                    )
                )

                new_duration = st.number_input(
                    "المدة بالدقائق",
                    min_value=1,
                    value=max(1, int(task["estimated_minutes"]))
                )

                save_edit = st.form_submit_button("حفظ التعديل")

            if save_edit:

                if not new_name.strip():

                    st.error("اكتب اسم المهمة.")

                else:

                    idx = tasks.index[
                        tasks["task_id"] == task_id
                    ]

                    tasks.loc[idx, "task_name"] = new_name.strip()
                    tasks.loc[idx, "category"] = new_category
                    tasks.loc[idx, "priority"] = new_priority
                    tasks.loc[idx, "due_date"] = new_due.isoformat()
                    tasks.loc[idx, "estimated_minutes"] = new_duration

                    save_tasks(tasks)

                    del st.session_state["editing_task"]
                    st.rerun()

            if st.button("إلغاء التعديل"):

                del st.session_state["editing_task"]
                st.rerun()


# ==================================================
# الصفحة الثانية: إضافة مهمة
# ==================================================

elif page == "إضافة مهمة":

    st.header("إضافة مهمة جديدة")

    with st.form("add_task_form"):

        task_name = st.text_input("اسم المهمة")

        category = st.selectbox(
            "التصنيف",
            list(CATEGORIES),
            format_func=lambda x: CATEGORIES[x]
        )

        priority = st.selectbox(
            "الأولوية",
            list(PRIORITIES),
            format_func=lambda x: PRIORITIES[x]
        )

        due_date = st.date_input(
            "تاريخ الاستحقاق",
            value=date.today()
        )

        days = st.number_input(
            "المدة بالأيام",
            min_value=0,
            value=0
        )

        hours = st.number_input(
            "المدة بالساعات",
            min_value=0,
            max_value=23,
            value=1
        )

        minutes = st.number_input(
            "المدة بالدقائق",
            min_value=0,
            max_value=59,
            value=0
        )

        add_button = st.form_submit_button(
            "إضافة المهمة",
            use_container_width=True
        )

    if add_button:

        estimated_minutes = (
            int(days) * 1440 +
            int(hours) * 60 +
            int(minutes)
        )

        if not task_name.strip():

            st.error("اكتب اسم المهمة أولًا.")

        elif estimated_minutes == 0:

            st.error("أدخل مدة متوقعة للمهمة.")

        else:

            new_id = (
                int(tasks["task_id"].max()) + 1
                if tasks["task_id"].notna().any()
                else 1
            )

            new_task = {
                "task_id": new_id,
                "task_name": task_name.strip(),
                "created_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "due_date": due_date.isoformat(),
                "estimated_minutes": estimated_minutes,
                "status": "incomplete",
                "completed_date": "",
                "category": category,
                "priority": priority,
                "deleted_date": ""
            }

            tasks = pd.concat(
                [tasks, pd.DataFrame([new_task])],
                ignore_index=True
            )

            save_tasks(tasks)

            st.success("تمت إضافة المهمة بنجاح.")
            st.rerun()


# ==================================================
# الصفحة الثالثة: سلة المحذوفات
# ==================================================

elif page == "سلة المحذوفات":

    st.header("سلة المحذوفات")

    if deleted_tasks.empty:

        st.info("سلة المحذوفات فارغة.")

    else:

        for _, task in deleted_tasks.iterrows():

            task_id = int(task["task_id"])

            with st.container(border=True):

                st.subheader(str(task["task_name"]))

                col1, col2 = st.columns(2)

                if col1.button(
                    "استعادة",
                    key=f"restore_{task_id}"
                ):

                    idx = tasks.index[
                        tasks["task_id"] == task_id
                    ]

                    tasks.loc[idx, "deleted_date"] = ""

                    save_tasks(tasks)
                    st.rerun()

                if col2.button(
                    "حذف نهائي",
                    key=f"permanent_{task_id}"
                ):

                    tasks = tasks[
                        tasks["task_id"] != task_id
                    ]

                    save_tasks(tasks)
                    st.rerun()


# ==================================================
# الصفحة الرابعة: التحليلات الشخصية
# ==================================================

elif page == "تحليل البيانات":

    st.header("التحليلات الشخصية")

   

    if active_tasks.empty:

        st.info("أضف مهامًا حتى تظهر التحليلات.")

    else:

        analysis_df, _ = prepare_analysis_data(active_tasks)

        valid = analysis_df.dropna(
            subset=["target"]
        ).copy()

        total = len(valid)

        completed = int(valid["target"].sum())

        rate = (
            completed / total * 100
            if total else 0
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("المهام المسجلة", total)
        c2.metric("المهام المكتملة", completed)
        c3.metric("نسبة الإنجاز", f"{rate:.1f}%")

        st.progress(
            int(rate),
            text=f"نسبة الإنجاز: {rate:.1f}%"
        )

        category_analysis = (
            valid.groupby("category")["target"]
            .mean()
            .mul(100)
            .dropna()
        )

        if not category_analysis.empty:

            st.subheader("نسبة الإنجاز حسب التصنيف")
            st.bar_chart(category_analysis)

        priority_analysis = (
            valid.groupby("priority")["target"]
            .mean()
            .mul(100)
            .dropna()
        )

        if not priority_analysis.empty:

            st.subheader("نسبة الإنجاز حسب الأولوية")
            st.bar_chart(priority_analysis)

        completed_df = valid[
            valid["target"] == 1
        ].dropna(
            subset=["completed_date"]
        ).copy()

        if not completed_df.empty:

            day_names = {
                "Monday": "الاثنين",
                "Tuesday": "الثلاثاء",
                "Wednesday": "الأربعاء",
                "Thursday": "الخميس",
                "Friday": "الجمعة",
                "Saturday": "السبت",
                "Sunday": "الأحد"
            }

            completed_df["day_name"] = (
                completed_df["completed_date"]
                .dt.day_name()
                .map(day_names)
            )

            daily = completed_df["day_name"].value_counts()

            st.subheader("أيام تسجيل الإنجاز")
            st.bar_chart(daily)


# ==================================================
# الصفحة الخامسة: التوقعات والتوصيات
# ==================================================

elif page == "التوقعات والتوصيات":

    st.header("التوقعات والتوصيات")

   

    try:

        training_data = pd.read_csv(
            TRAINING_FILE,
            keep_default_na=False
        )

    except FileNotFoundError:

        st.error(
            "ملف training_tasks.csv غير موجود. "
            "أضيفيه إلى المستودع بجانب app.py."
        )

        st.stop()

    except pd.errors.EmptyDataError:

        st.error(
            "ملف training_tasks.csv فارغ. "
            "تأكدي من رفع بيانات التدريب الصحيحة."
        )

        st.stop()

    required = [
        "task_id",
        "created_date",
        "due_date",
        "estimated_minutes",
        "status",
        "completed_date",
        "category",
        "priority"
    ]

    missing = [
        column for column in required
        if column not in training_data.columns
    ]

    if missing:

        st.error(
            "أعمدة ناقصة في ملف التدريب: "
            + ", ".join(missing)
        )

        st.stop()

    training_df, _ = prepare_analysis_data(training_data)

    training_df = training_df.dropna(
        subset=FEATURES + ["target"]
    ).copy()

    if len(training_df) < 10:

        st.warning(
            "عدد السجلات الصالحة في بيانات التدريب أقل من 10."
        )

        st.stop()

    if training_df["target"].nunique() < 2:

        st.warning(
            "ملف التدريب يجب أن يحتوي على مهام مكتملة وغير مكتملة."
        )

        st.stop()

    if training_df["target"].value_counts().min() < 2:

        st.warning(
            "عدد إحدى فئات الإنجاز قليل جدًا للتدريب والاختبار."
        )

        st.stop()

    X = training_df[FEATURES].astype(float)
    y = training_df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = accuracy_score(
        y_test,
        model.predict(X_test)
    ) * 100

    c1, c2 = st.columns(2)

    c1.metric(
        "سجلات التدريب المستخدمة",
        len(training_df)
    )

    c2.metric(
        "دقة النموذج الاختبارية",
        f"{accuracy:.1f}%"
    )

    st.caption(
        "الدقة تقيس أداء النموذج على عينة اختبار تاريخية، "
        "ولا تضمن إنجاز مهمة مستقبلية."
    )

    # العوامل المؤثرة

    st.subheader("أهم العوامل المؤثرة على الإنجاز")

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES
    )

    importance["category"] = importance[
        [
            "cat_study",
            "cat_home",
            "cat_exercise",
            "cat_personal"
        ]
    ].sum()

    importance = importance.drop([
        "cat_study",
        "cat_home",
        "cat_exercise",
        "cat_personal"
    ])

    labels = {
        "weekday": "يوم إضافة المهمة",
        "hour": "ساعة إضافة المهمة",
        "days_to_due": "الأيام المتبقية للموعد",
        "tasks_same_day": "عدد مهام اليوم",
        "priority_num": "الأولوية",
        "estimated_minutes": "المدة المتوقعة",
        "category": "التصنيف"
    }

    importance.index = [
        labels.get(x, x)
        for x in importance.index
    ]

    st.bar_chart(
        importance.sort_values(ascending=False) * 100
    )

    st.divider()

    # توقع مهمة جديدة

    st.subheader("توقع مهمة جديدة")

    with st.form("prediction_form"):

        weekdays = [
            "الاثنين",
            "الثلاثاء",
            "الأربعاء",
            "الخميس",
            "الجمعة",
            "السبت",
            "الأحد"
        ]

        task_day = st.selectbox(
            "يوم إضافة المهمة",
            weekdays
        )

        task_hour = st.slider(
            "ساعة إضافة المهمة",
            0,
            23,
            9
        )

        days_left = st.number_input(
            "الأيام المتبقية على الموعد",
            0,
            365,
            2
        )

        same_day = st.number_input(
            "عدد مهام اليوم",
            1,
            100,
            3
        )

        priority_choice = st.selectbox(
            "الأولوية",
            ["عالية", "متوسطة", "منخفضة"]
        )

        category_choice = st.selectbox(
            "التصنيف",
            list(CATEGORIES),
            format_func=lambda x: CATEGORIES[x]
        )

        duration = st.number_input(
            "المدة المتوقعة بالدقائق",
            1,
            100000,
            60
        )

        predict_button = st.form_submit_button(
            "توقع احتمال الإنجاز",
            use_container_width=True
        )

    if predict_button:

        day_values = {
            name: i
            for i, name in enumerate(weekdays)
        }

        priority_values = {
            "منخفضة": 1,
            "متوسطة": 2,
            "عالية": 3
        }

        new_row = {
            "weekday": day_values[task_day],
            "hour": task_hour,
            "days_to_due": days_left,
            "tasks_same_day": same_day,
            "priority_num": priority_values[priority_choice],
            "estimated_minutes": duration,
            "cat_study": int(category_choice == "study"),
            "cat_home": int(category_choice == "home"),
            "cat_exercise": int(category_choice == "exercise"),
            "cat_personal": int(category_choice == "personal")
        }

        new_task = pd.DataFrame(
            [new_row],
            columns=FEATURES
        )

        probability = model.predict_proba(new_task)[0]

        complete_index = list(
            model.classes_
        ).index(1)

        completion_probability = (
            probability[complete_index] * 100
        )

        st.subheader("نتيجة التوقع")

        st.metric(
            "احتمال إنجاز المهمة",
            f"{completion_probability:.1f}%"
        )

        st.progress(
            int(round(completion_probability))
        )

        if completion_probability >= 70:

            st.success(
                "احتمال الإنجاز مرتفع وفقًا للبيانات التاريخية."
            )

        elif completion_probability >= 40:

            st.warning(
                "احتمال الإنجاز متوسط. "
                "قد يفيدك تنظيم الوقت وتقسيم المهمة."
            )

        else:

            st.warning(
                "احتمال الإنجاز منخفض وفقًا للبيانات التاريخية. "
                "جرّب تقسيم المهمة."
            )

    st.divider()

    # توصيات الفريق

    st.subheader("توصيات مستخلصة من بيانات الفريق")

    day_names = {
        0: "الاثنين",
        1: "الثلاثاء",
        2: "الأربعاء",
        3: "الخميس",
        4: "الجمعة",
        5: "السبت",
        6: "الأحد"
    }

    day_rates = (
        training_df.groupby("weekday")["target"]
        .mean()
        .dropna()
    )

    if not day_rates.empty:

        st.info(
            "أعلى يوم إنجازًا في بيانات الفريق: "
            + day_names[int(day_rates.idxmax())]
        )

    period_df = training_df.copy()

    period_df["period"] = pd.cut(
        period_df["hour"],
        bins=[-1, 11, 17, 23],
        labels=["الصباح", "الظهر", "المساء"]
    )

    period_rates = (
        period_df.groupby(
            "period",
            observed=False
        )["target"].mean().dropna()
    )

    if not period_rates.empty:

        st.info(
            f"أفضل فترة حسب البيانات: {period_rates.idxmax()}"
        )

    category_rates = (
        training_df.groupby("category")["target"]
        .mean()
        .dropna()
    )

    if not category_rates.empty:

        category_names = {
            "study": "الدراسة",
            "home": "البيت",
            "exercise": "الرياضة",
            "personal": "الشخصي"
        }

        lowest = category_rates.idxmin()

        st.info(
            "التصنيف الأقل إنجازًا: "
            + category_names.get(lowest, lowest)
        )

    long_tasks = training_df[
        training_df["estimated_minutes"] >= 90
    ]

    short_tasks = training_df[
        training_df["estimated_minutes"] < 90
    ]

    if not long_tasks.empty and not short_tasks.empty:

        long_rate = long_tasks["target"].mean() * 100
        short_rate = short_tasks["target"].mean() * 100

        if long_rate < short_rate:

            st.warning(
                f"إنجاز المهام الطويلة (90 دقيقة فأكثر) بلغ "
                f"{long_rate:.1f}% مقابل "
                f"{short_rate:.1f}% للمهام الأقصر. "
                f"تقسيم المهمة قد يساعد."
            )
