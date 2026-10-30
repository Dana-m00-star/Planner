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

STATUS_NAMES = {
    "complete": "مكتملة",
    "incomplete": "غير مكتملة"
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


def load_tasks():
    try:
        df = pd.read_csv(FILE_NAME)
    except FileNotFoundError:
        df = pd.DataFrame(columns=COLUMNS)

    for column in COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df["task_id"] = pd.to_numeric(
        df["task_id"], errors="coerce"
    )

    df["estimated_minutes"] = pd.to_numeric(
        df["estimated_minutes"], errors="coerce"
    ).fillna(0)

    df["deleted_date"] = df["deleted_date"].fillna("")
    df["completed_date"] = df["completed_date"].fillna("")

    return df[COLUMNS]


def save_tasks(df):
    df.to_csv(FILE_NAME, index=False)


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

    df["target"] = (
        df["status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(status_map)
    )

    priority_map = {
        "low": 1,
        "منخفضة": 1,
        "medium": 2,
        "متوسطة": 2,
        "high": 3,
        "عالية": 3
    }

    df["priority_num"] = (
        df["priority"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(priority_map)
    )

    df["created_date"] = pd.to_datetime(
        df["created_date"], errors="coerce"
    )

    df["due_date"] = pd.to_datetime(
        df["due_date"], errors="coerce"
    )

    df["completed_date"] = pd.to_datetime(
        df["completed_date"], errors="coerce"
    )

    df["estimated_minutes"] = pd.to_numeric(
        df["estimated_minutes"], errors="coerce"
    ).fillna(0)

    df["weekday"] = df["created_date"].dt.weekday
    df["hour"] = df["created_date"].dt.hour

    df["days_to_due"] = (
        df["due_date"] - df["created_date"].dt.normalize()
    ).dt.days

    df["tasks_same_day"] = (
        df.groupby(df["created_date"].dt.date)["task_id"]
        .transform("count")
    )

    df["category"] = df["category"].astype(str)

    category_dummies = pd.get_dummies(
        df["category"], prefix="cat"
    ).astype(int)

    df = pd.concat(
        [df, category_dummies],
        axis=1
    )

    return df, category_dummies.columns.tolist()


tasks = load_tasks()

st.markdown("""
<style>
.stApp {
    background-color: #F3EBDD;
}
[data-testid="stSidebar"] {
    background-color: #192B49;
}
[data-testid="stSidebar"] * {
    color: #F3EBDD;
}
h1, h2, h3 {
    color: #192B49;
}
.stButton button {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("Planner 💙")
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

active_tasks = tasks[
    tasks["deleted_date"].astype(str).eq("")
].copy()

completed_count = (
    active_tasks["status"].astype(str) == "complete"
).sum()

incomplete_count = len(active_tasks) - completed_count

completion_rate = (
    completed_count / len(active_tasks) * 100
    if len(active_tasks) else 0
)

if page == "مهامي":

    st.header("مهامي")

    c1, c2, c3 = st.columns(3)

    c1.metric("إجمالي المهام", len(active_tasks))
    c2.metric("المهام المكتملة", int(completed_count))
    c3.metric("المهام المتبقية", int(incomplete_count))

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
        format_func=lambda x: "الكل" if x == "الكل" else CATEGORIES[x]
    )

    displayed_tasks = active_tasks.copy()

    if search:
        displayed_tasks = displayed_tasks[
            displayed_tasks["task_name"]
            .astype(str)
            .str.contains(search, case=False, na=False)
        ]

    if filter_status == "مكتملة":
        displayed_tasks = displayed_tasks[
            displayed_tasks["status"] == "complete"
        ]

    elif filter_status == "غير مكتملة":
        displayed_tasks = displayed_tasks[
            displayed_tasks["status"] == "incomplete"
        ]

    if filter_category != "الكل":
        displayed_tasks = displayed_tasks[
            displayed_tasks["category"] == filter_category
        ]

    if displayed_tasks.empty:
        st.info("لا توجد مهام لعرضها.")

    for _, task in displayed_tasks.iterrows():

        task_id = task["task_id"]

        with st.container(border=True):

            st.subheader(str(task["task_name"]))

            category_name = CATEGORIES.get(
                task["category"], task["category"]
            )

            priority_name = PRIORITIES.get(
                task["priority"], task["priority"]
            )

            st.write(f"التصنيف: {category_name}")
            st.write(f"الأولوية: {priority_name}")

            st.write(
                f"تاريخ الاستحقاق: {task['due_date']}"
            )

            st.write(
                f"المدة المتوقعة: {int(task['estimated_minutes'])} دقيقة"
            )

            is_complete = task["status"] == "complete"

            new_status = st.checkbox(
                "مكتملة",
                value=is_complete,
                key=f"complete_{task_id}"
            )

            if new_status != is_complete:

                index = tasks.index[
                    tasks["task_id"] == task_id
                ]

                if new_status:
                    tasks.loc[index, "status"] = "complete"
                    tasks.loc[index, "completed_date"] = (
                        datetime.now().strftime("%Y-%m-%d %H:%M")
                    )
                else:
                    tasks.loc[index, "status"] = "incomplete"
                    tasks.loc[index, "completed_date"] = ""

                save_tasks(tasks)
                st.rerun()

            col1, col2 = st.columns(2)

            if col1.button(
                "حذف المهمة",
                key=f"delete_{task_id}"
            ):

                index = tasks.index[
                    tasks["task_id"] == task_id
                ]

                tasks.loc[index, "deleted_date"] = (
                    datetime.now().strftime("%Y-%m-%d %H:%M")
                )

                save_tasks(tasks)
                st.rerun()

            if col2.button(
                "تعديل المهمة",
                key=f"edit_{task_id}"
            ):

                st.session_state["editing_task"] = int(task_id)
                st.rerun()

    if "editing_task" in st.session_state:

        task_id = st.session_state["editing_task"]

        selected = tasks[
            tasks["task_id"] == task_id
        ]

        if not selected.empty:

            task = selected.iloc[0]

            st.divider()
            st.subheader("تعديل المهمة")

            with st.form("edit_task_form"):

                new_name = st.text_input(
                    "اسم المهمة",
                    value=str(task["task_name"])
                )

                new_category = st.selectbox(
                    "التصنيف",
                    list(CATEGORIES.keys()),
                    index=list(CATEGORIES.keys()).index(task["category"])
                    if task["category"] in CATEGORIES else 0,
                    format_func=lambda x: CATEGORIES[x]
                )

                new_priority = st.selectbox(
                    "الأولوية",
                    list(PRIORITIES.keys()),
                    index=list(PRIORITIES.keys()).index(task["priority"])
                    if task["priority"] in PRIORITIES else 0,
                    format_func=lambda x: PRIORITIES[x]
                )

                new_due = st.date_input(
                    "تاريخ الاستحقاق",
                    value=pd.to_datetime(
                        task["due_date"], errors="coerce"
                    ).date()
                    if pd.notna(pd.to_datetime(
                        task["due_date"], errors="coerce"
                    )) else date.today()
                )

                new_duration = st.number_input(
                    "المدة بالدقائق",
                    min_value=1,
                    value=max(1, int(task["estimated_minutes"]))
                )

                save_edit = st.form_submit_button("حفظ التعديل")

            if save_edit:

                index = tasks.index[
                    tasks["task_id"] == task_id
                ]

                tasks.loc[index, "task_name"] = new_name
                tasks.loc[index, "category"] = new_category
                tasks.loc[index, "priority"] = new_priority
                tasks.loc[index, "due_date"] = new_due.strftime("%Y-%m-%d")
                tasks.loc[index, "estimated_minutes"] = new_duration

                save_tasks(tasks)

                del st.session_state["editing_task"]
                st.rerun()

            if st.button("إلغاء التعديل"):
                del st.session_state["editing_task"]
                st.rerun()


elif page == "إضافة مهمة":

    st.header("إضافة مهمة جديدة")

    with st.form("add_task_form"):

        task_name = st.text_input("اسم المهمة")

        category = st.selectbox(
            "التصنيف",
            list(CATEGORIES.keys()),
            format_func=lambda x: CATEGORIES[x]
        )

        priority = st.selectbox(
            "الأولوية",
            list(PRIORITIES.keys()),
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

        if not task_name.strip():
            st.error("اكتب اسم المهمة أولًا.")

        elif days == 0 and hours == 0 and minutes == 0:
            st.error("أدخل مدة متوقعة للمهمة.")

        else:

            new_id = (
                int(tasks["task_id"].max()) + 1
                if tasks["task_id"].notna().any()
                else 1
            )

            estimated_minutes = (
                int(days) * 1440
                + int(hours) * 60
                + int(minutes)
            )

            new_task = {
                "task_id": new_id,
                "task_name": task_name.strip(),
                "created_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "due_date": due_date.strftime("%Y-%m-%d"),
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


elif page == "سلة المحذوفات":

    st.header("سلة المحذوفات")

    deleted_tasks = tasks[
        tasks["deleted_date"].astype(str).ne("")
    ].copy()

    if deleted_tasks.empty:

        st.info("سلة المحذوفات فارغة.")

    else:

        for _, task in deleted_tasks.iterrows():

            task_id = task["task_id"]

            with st.container(border=True):

                st.subheader(str(task["task_name"]))

                col1, col2 = st.columns(2)

                if col1.button(
                    "استعادة",
                    key=f"restore_{task_id}"
                ):

                    index = tasks.index[
                        tasks["task_id"] == task_id
                    ]

                    tasks.loc[index, "deleted_date"] = ""

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


elif page == "تحليل البيانات":

    st.header("التحليلات الذكية")

    st.write(
        "تحليل إنجاز المهام والتعرف على أنماط العمل."
    )

    if active_tasks.empty:

        st.info("أضف مهامًا حتى تظهر التحليلات.")

    else:

        df_analysis, category_columns = prepare_analysis_data(
            active_tasks
        )

        total = len(df_analysis)
        completed = int(
            df_analysis["target"].fillna(0).sum()
        )
        incomplete = total - completed

        completion_rate = (
            completed / total * 100 if total else 0
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("إجمالي المهام", total)
        c2.metric("المهام المنجزة", completed)
        c3.metric("المهام المتبقية", incomplete)
        c4.metric("نسبة الإنجاز", f"{completion_rate:.1f}%")

        st.divider()

        st.subheader("نسبة إنجاز المهام")

        st.progress(
            int(completion_rate),
            text=f"{completion_rate:.1f}%"
        )

        st.subheader("إنجاز المهام حسب التصنيف")

        category_analysis = (
            df_analysis.groupby("category")["target"]
            .mean()
            .mul(100)
            .dropna()
            .sort_values(ascending=False)
        )

        if not category_analysis.empty:

            st.bar_chart(category_analysis)

            st.success(
                f"أعلى تصنيف في الإنجاز: {category_analysis.idxmax()}"
            )

        st.divider()

        st.subheader("أكثر الأيام إنجازًا")

        completed_df = df_analysis[
            df_analysis["target"] == 1
        ].copy()

        completed_df["day_name"] = (
            completed_df["completed_date"].dt.day_name()
        )

        day_names = {
            "Monday": "الاثنين",
            "Tuesday": "الثلاثاء",
            "Wednesday": "الأربعاء",
            "Thursday": "الخميس",
            "Friday": "الجمعة",
            "Saturday": "السبت",
            "Sunday": "الأحد"
        }

        daily_completion = (
            completed_df["day_name"]
            .value_counts()
            .rename(index=day_names)
        )

        if not daily_completion.empty:
            st.bar_chart(daily_completion)
            st.info(
                f"أكثر يوم إنجازًا: {daily_completion.idxmax()}"
            )
        else:
            st.info("لا توجد تواريخ إنجاز مسجلة.")

        st.divider()

        st.subheader("تحليل الأولويات")

        priority_analysis = (
            df_analysis.groupby("priority")["target"]
            .mean()
            .mul(100)
            .dropna()
        )

        if not priority_analysis.empty:
            st.bar_chart(priority_analysis)

        st.subheader("العلاقة بين مدة المهمة والإنجاز")

        duration_analysis = (
            df_analysis.groupby("estimated_minutes")["target"]
            .mean()
            .mul(100)
            .dropna()
        )

        if not duration_analysis.empty:
            st.line_chart(duration_analysis)

        st.divider()

        st.subheader("الالتزام بالمواعيد")

        on_time_df = completed_df.dropna(
            subset=["completed_date", "due_date"]
        ).copy()

        if not on_time_df.empty:

            on_time_df["on_time"] = (
                on_time_df["completed_date"].dt.normalize()
                <= on_time_df["due_date"]
            )

            on_time_rate = (
                on_time_df["on_time"].mean() * 100
            )

            late_rate = 100 - on_time_rate

            x, y = st.columns(2)

            x.metric(
                "الإنجاز في الموعد",
                f"{on_time_rate:.1f}%"
            )

            y.metric(
                "الإنجاز المتأخر",
                f"{late_rate:.1f}%"
            )

        else:
            st.info("لا توجد بيانات كافية لتحليل المواعيد.")


elif page == "التوقعات والتوصيات":

    st.header("التوقعات والتوصيات")

    st.write(
        "يتعلم النموذج من المهام السابقة لتقدير احتمال إنجاز مهمة جديدة."
    )

    ml_tasks = active_tasks.copy()

    if len(ml_tasks) < 10:

        st.warning(
            "نحتاج إلى 10 مهام على الأقل لبدء تدريب النموذج."
        )

    else:

        ml_df, ml_category_columns = prepare_analysis_data(
            ml_tasks
        )

        ml_df = ml_df.dropna(
            subset=[
                "target",
                "weekday",
                "hour",
                "days_to_due",
                "priority_num"
            ]
        ).copy()

        features = [
            "weekday",
            "hour",
            "days_to_due",
            "tasks_same_day",
            "priority_num",
            "estimated_minutes"
        ] + ml_category_columns

        if ml_df["target"].nunique() < 2:

            st.warning(
                "يحتاج النموذج إلى مهام منجزة وغير منجزة حتى يتعلم."
            )

        elif len(ml_df) < 10:

            st.warning("البيانات الصالحة للتدريب غير كافية.")

        else:

            X = ml_df[features].fillna(0)
            y = ml_df["target"].astype(int)

            stratify_value = (
                y if y.value_counts().min() >= 2 else None
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=stratify_value
            )

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=42
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions
            ) * 100

            st.success("تم تدريب النموذج بنجاح.")

            st.metric(
                "دقة النموذج على بيانات الاختبار",
                f"{accuracy:.1f}%"
            )

            st.caption(
                "الدقة تتأثر بعدد المهام وتوازن حالات الإنجاز وجودة البيانات."
            )

            st.divider()

            st.subheader("أهم العوامل المؤثرة على الإنجاز")

            importance = pd.Series(
                model.feature_importances_,
                index=features
            )

            category_importance = importance[
                ml_category_columns
            ].sum()

            importance = importance.drop(
                ml_category_columns
            )

            importance["category"] = category_importance

            feature_names = {
                "weekday": "يوم إضافة المهمة",
                "hour": "ساعة إضافة المهمة",
                "days_to_due": "الأيام المتبقية للموعد",
                "tasks_same_day": "عدد مهام اليوم",
                "priority_num": "الأولوية",
                "estimated_minutes": "المدة المتوقعة",
                "category": "تصنيف المهمة"
            }

            importance.index = [
                feature_names.get(name, name)
                for name in importance.index
            ]

            importance = (
                importance.sort_values(ascending=False) * 100
            )

            st.bar_chart(importance)

            st.divider()

            st.subheader("توقع مهمة جديدة")

            with st.form("prediction_form"):

                task_day = st.selectbox(
                    "يوم إضافة المهمة",
                    [
                        "الاثنين",
                        "الثلاثاء",
                        "الأربعاء",
                        "الخميس",
                        "الجمعة",
                        "السبت",
                        "الأحد"
                    ]
                )

                day_values = {
                    "الاثنين": 0,
                    "الثلاثاء": 1,
                    "الأربعاء": 2,
                    "الخميس": 3,
                    "الجمعة": 4,
                    "السبت": 5,
                    "الأحد": 6
                }

                task_hour = st.slider(
                    "ساعة إضافة المهمة",
                    0,
                    23,
                    9
                )

                days_left = st.number_input(
                    "الأيام المتبقية على الموعد",
                    min_value=0,
                    max_value=365,
                    value=2
                )

                same_day = st.number_input(
                    "عدد المهام في نفس اليوم",
                    min_value=1,
                    max_value=100,
                    value=3
                )

                priority_choice = st.selectbox(
                    "الأولوية",
                    ["عالية", "متوسطة", "منخفضة"]
                )

                priority_values = {
                    "منخفضة": 1,
                    "متوسطة": 2,
                    "عالية": 3
                }

                category_choice = st.selectbox(
                    "التصنيف",
                    list(CATEGORIES.keys()),
                    format_func=lambda x: CATEGORIES[x]
                )

                duration = st.number_input(
                    "المدة المتوقعة بالدقائق",
                    min_value=1,
                    max_value=100000,
                    value=60
                )

                predict_button = st.form_submit_button(
                    "توقع احتمال الإنجاز",
                    use_container_width=True
                )

            if predict_button:

                new_row = {
                    "weekday": day_values[task_day],
                    "hour": task_hour,
                    "days_to_due": days_left,
                    "tasks_same_day": same_day,
                    "priority_num": priority_values[priority_choice],
                    "estimated_minutes": duration
                }

                for col in ml_category_columns:
                    new_row[col] = int(
                        col == "cat_" + category_choice
                    )

                new_task = pd.DataFrame(
                    [new_row]
                )[features]

                probability = model.predict_proba(
                    new_task
                )[0]

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
                        "احتمال الإنجاز مرتفع حسب البيانات السابقة."
                    )

                elif completion_probability >= 40:

                    st.warning(
                        "احتمال الإنجاز متوسط. حاول تنظيم وقت المهمة."
                    )

                else:

                    st.error(
                        "احتمال الإنجاز منخفض حسب البيانات السابقة. "
                        "قد يساعدك تقسيم المهمة إلى خطوات أصغر."
                    )

            st.divider()

            st.subheader("التوصيات")

            day_names_ar = {
                0: "الاثنين",
                1: "الثلاثاء",
                2: "الأربعاء",
                3: "الخميس",
                4: "الجمعة",
                5: "السبت",
                6: "الأحد"
            }

            by_day = (
                ml_df.groupby("weekday")["target"]
                .mean()
                .dropna()
            )

            if not by_day.empty:

                best_day = int(by_day.idxmax())

                st.info(
                    f"أفضل يوم للإنجاز حسب بياناتك: "
                    f"{day_names_ar[best_day]}"
                )

            def get_period(hour):

                if hour < 12:
                    return "الصباح"

                elif hour < 18:
                    return "الظهر"

                return "المساء"

            period_df = ml_df.copy()

            period_df["period"] = (
                period_df["hour"].apply(get_period)
            )

            period_rates = (
                period_df.groupby("period")["target"]
                .mean()
                .dropna()
            )

            if not period_rates.empty:

                st.info(
                    f"أفضل فترة للإنجاز: {period_rates.idxmax()}"
                )

            category_rates = (
                ml_df.groupby("category")["target"]
                .mean()
                .dropna()
            )

            if not category_rates.empty:

                st.info(
                    f"التصنيف الأقل إنجازًا: {category_rates.idxmin()}"
                )

            long_tasks = ml_df[
                ml_df["estimated_minutes"] >= 90
            ]

            short_tasks = ml_df[
                ml_df["estimated_minutes"] < 90
            ]

            if not long_tasks.empty and not short_tasks.empty:

                long_rate = (
                    long_tasks["target"].mean() * 100
                )

                short_rate = (
                    short_tasks["target"].mean() * 100
                )

                if long_rate < short_rate:

                    st.warning(
                        f"المهام التي مدتها 90 دقيقة أو أكثر "
                        f"نسبة إنجازها {long_rate:.1f}%، "
                        f"مقارنة بـ {short_rate:.1f}% للمهام الأقصر. "
                        f"قد يساعدك تقسيم المهام الطويلة."
                    )
