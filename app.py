"""Study office: who should the advisers talk to this week? (Streamlit app on top of a dropout-risk model)

The model is read from model/ (booster.json + preprocess.json, written by the notebook, loaded by portable.py).
If model/ is empty, the app trains the same XGBoost model itself on first start, so it always runs.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from portable import Model

URL = "https://raw.githubusercontent.com/aaubs/ds-master/main/assignments/study-office/data/"
MODEL_DIR = Path(__file__).parent / "model"
TALK_G, WORRY_G, LEAVE_G, HELPS_G = 500, 2000, 60000, 0.30   # costs used in the game
LEAKS = ["student_id", "cohort", "left", "ects_passed_sem1", "deregistration_form_opened", "last_login_week"]
NICE = {"logins_total": "logins weeks 1-6", "logins_last3": "logins last 3 weeks", "logins_trend": "login trend",
        "submitted_share": "assignments handed in", "missed_last3": "assignments missed (last 3 wks)",
        "quiz_mean": "average quiz score", "weeks_since_login": "weeks since last login", "fees_owed": "fees owed",
        "su_scholarship": "SU scholarship", "admission_grade": "admission grade", "first_gen": "first in family",
        "international": "international", "moved_from_home": "moved from home", "evening_programme": "evening programme"}

st.set_page_config(page_title="Study office: who to call", page_icon="🎓", layout="wide")


@st.cache_data
def data():
    return pd.read_csv(URL + "history_week6.csv"), pd.read_csv(URL + "new_week6.csv")


@st.cache_resource
def load_model():
    history, _ = data()
    if (MODEL_DIR / "booster.json").exists():
        return Model(MODEL_DIR), "stored model in model/"
    import xgboost as xgb, portable
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    feats = [c for c in history.columns if c not in LEAKS]
    cat = [c for c in ["programme", "gender"] if c in feats]
    num = [c for c in feats if c not in cat]
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False), cat)])
    pipe = make_pipeline(prep, xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3))
    tr = history[history["cohort"] <= 2024]
    pipe.fit(tr[feats], tr["left"])
    tmp = tempfile.mkdtemp()
    portable.export(pipe, tmp)
    return Model(tmp), "trained at start-up on 2023-2024 (no model/ folder found)"


def boxes(y, flag):
    y, flag = np.asarray(y) == 1, np.asarray(flag)
    tp, fp, fn, tn = int((flag & y).sum()), int((flag & ~y).sum()), int((~flag & y).sum()), int((~flag & ~y).sum())
    return dict(reached=tp, worried=fp, missed=fn, fine=tn, contacted=tp + fp,
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1))


def show_boxes(b, label=""):
    st.markdown(f"{label}**{b['reached']} students reached in time, {b['worried']} worried for nothing, "
                f"{b['missed']} missed.** ({b['fine']} stayed and were left alone.)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Conversations", b["contacted"])
    c2.metric("Precision", f"{b['precision']:.0%}", help="Of the students we talked to, the share who really were at risk.")
    c3.metric("Recall", f"{b['recall']:.0%}", help="Of the students who left, the share we talked to.")


def reasons(model, rows, top=3):
    contrib = model.contributions(rows[model.features]).reset_index(drop=True)
    out = []
    for i, (_, r) in enumerate(rows.iterrows()):
        best = contrib.iloc[i].sort_values(ascending=False).head(top)
        out.append(" · ".join(f"{NICE.get(f, f.replace('_', ' '))}: {r[f]:.2g}" if isinstance(r[f], (int, float, np.number))
                              else f"{f}: {r[f]}" for f, c in best.items() if c > 0))
    return out


model, source = load_model()
history, new = data()
val = history[history["cohort"] == 2025].copy()
val["risk"] = model.predict_proba(val[model.features])
new["risk"] = model.predict_proba(new[model.features])

with st.sidebar:
    st.header("🛠️ The rule")
    mode = st.radio("Decide by", ["Number of conversations", "Risk cut-off"])
    if mode == "Number of conversations":
        n = st.slider("Conversations at week 6", 5, 150, 40, 5, help="Three advisers: about 40.")
        cut_new = new["risk"].nlargest(n).min()
        flag_val = (val["risk"].rank(ascending=False, method="first") <= n).to_numpy()
        flag_new = (new["risk"] >= cut_new).to_numpy()
    else:
        cut = st.slider("Contact students with risk at least", 0.02, 0.90, 0.30, 0.01, format="%.2f")
        flag_val, flag_new = (val["risk"] >= cut).to_numpy(), (new["risk"] >= cut).to_numpy()
    st.caption(f"Model: {source}. Learned from 2023-2024, checked on 2025 ({len(val)} students).")

st.title("🎓 Who should the study office talk to this week?")
t_list, t_rule, t_group, t_cost, t_game = st.tabs(["📋 This week's list", "⚖️ Mistakes of the rule",
                                                   "🌍 Fair to whom?", "💶 Costs & best rule", "🎲 Play: be the adviser"])

with t_list:
    st.markdown(f"All **{len(new)}** students of this year, ranked by risk of leaving. "
                f"**{int(flag_new.sum())}** are marked for a conversation with the current rule.")
    lst = new.assign(talk=flag_new).sort_values("risk", ascending=False).reset_index(drop=True)
    top = lst.head(max(int(flag_new.sum()), 1)).copy()
    lst["why"] = ""
    lst.loc[top.index, "why"] = reasons(model, top)   # explanation for the marked students
    st.dataframe(lst[["talk", "student_id", "risk", "programme", "why"]],
                 column_config={"talk": st.column_config.CheckboxColumn("talk to"),
                                "risk": st.column_config.ProgressColumn("risk of leaving", min_value=0, max_value=1, format="%.2f"),
                                "why": "what pushes the risk up (marked students)"},
                 hide_index=True, width="stretch", height=520)
    st.download_button("⬇️ Download the list (CSV)", lst.to_csv(index=False), "talk_to_this_week.csv")

with t_rule:
    st.markdown("What would this rule have done to the **2025 cohort**, where we know who left?")
    show_boxes(boxes(val["left"], flag_val))
    base = val["left"].mean()
    st.info(f"{base:.0%} of the 2025 students left. Contacting nobody would be right for {1 - base:.0%} of students, "
            "and still miss every student who leaves: that is why accuracy is not shown.")

with t_group:
    col = st.selectbox("Compare", ["international", "first_gen", "gender", "evening_programme"])
    st.markdown("Same rule, same 2025 cohort, split by group.")
    cols = st.columns(val[col].nunique())
    for c, (g, d) in zip(cols, val.assign(flag=flag_val).groupby(col)):
        with c:
            st.subheader(f"{col} = {g}")
            st.caption(f"{len(d)} students · {d['left'].mean():.0%} left · average predicted risk {d['risk'].mean():.0%}")
            show_boxes(boxes(d["left"], d["flag"]))
    st.caption("If one group has a clearly lower recall, the model misses more of its students: check whether the signals "
               "(e.g. logins) mean the same thing for everyone.")

with t_cost:
    st.markdown("**Your own costs decide the best rule.** Change the assumptions and see where the best cut-off moves.")
    a, b, c, d = st.columns(4)
    talk = a.number_input("A conversation (DKK)", 0, 5000, 500, 100)
    worry = b.number_input("A false alarm (DKK)", 0, 20000, 2000, 500)
    leave = c.number_input("A student who leaves (DKK)", 0, 300000, 60000, 5000)
    helps = d.slider("Conversation keeps (%)", 0, 100, 30) / 100
    cuts = np.round(np.arange(0.02, 0.91, 0.01), 2)
    value = []
    for t in cuts:
        k = boxes(val["left"], (val["risk"] >= t).to_numpy())
        value.append(k["reached"] * helps * leave - k["contacted"] * talk - k["worried"] * worry)
    best = int(np.argmax(value))
    k = boxes(val["left"], (val["risk"] >= cuts[best]).to_numpy())
    st.success(f"Best cut-off: **{cuts[best]:.2f}** → {k['contacted']} conversations, net value about {value[best]:,.0f} DKK on 2025.")
    st.line_chart(pd.DataFrame({"cut-off": cuts, "net value (DKK)": value}), x="cut-off", y="net value (DKK)")
    st.caption(f"With capacity for only {40} conversations, the cost-optimal rule may be out of reach: compare with the sidebar.")

# ---------------------------------------------------------------- the game
with t_game:
    CAP, SIZE = 5, 25
    ss = st.session_state
    ss.setdefault("round", 0); ss.setdefault("done", False); ss.setdefault("totals", {"you": 0.0, "model": 0.0, "gus": 0.0})
    r = ss.round
    batch = val.sample(SIZE, random_state=r + 1).reset_index(drop=True)
    st.markdown(f"It is week 6. **{SIZE} students** from the 2025 cohort are on your desk, and you only have time for "
                f"**{CAP} conversations**. Tick the ones you would talk to. Afterwards you see who really left, and how you did "
                "against the model and against **Gut-Feeling Gus**, who simply calls the students who hand in the least.")
    hints = st.toggle("Show the model's risk", value=True, help="Hard mode: off.")
    show = batch[["student_id", "programme", "submitted_share", "logins_last3", "quiz_mean", "weeks_since_login", "fees_owed"]].copy()
    if hints:
        show["model risk"] = batch["risk"].round(2)
    show.insert(0, "talk", False)
    edited = st.data_editor(show, hide_index=True, width="stretch", key=f"ed{r}",
                            disabled=[c for c in show.columns if c != "talk"] if not ss.done else True)
    mine = edited["talk"].to_numpy()
    st.caption(f"Selected: {int(mine.sum())} of {CAP}")
    if mine.sum() > CAP:
        st.error(f"Only {CAP} conversations are possible. Untick {int(mine.sum()) - CAP}.")
    if not ss.done:
        if st.button("🚪 Reveal who left", type="primary", disabled=bool(mine.sum() > CAP)):
            ss.done = True
            st.rerun()
    else:
        y = batch["left"].to_numpy()
        flags = {"you": mine,
                 "model": (batch["risk"].rank(ascending=False, method="first") <= CAP).to_numpy(),
                 "gus": (batch["submitted_share"].rank(method="first") <= CAP).to_numpy()}
        res = {}
        for who, f in flags.items():
            b = boxes(y, f)
            res[who] = (b["reached"] * HELPS_G * LEAVE_G - b["contacted"] * TALK_G - b["worried"] * WORRY_G, b)
        if "counted" not in ss or ss.counted != r:
            for who in res:
                ss.totals[who] += res[who][0]
            ss.counted = r
        c1, c2, c3 = st.columns(3)
        for c, (who, name) in zip((c1, c2, c3), [("you", "You"), ("model", "The model"), ("gus", "Gut-Feeling Gus")]):
            b = res[who][1]
            c.metric(name, f"{res[who][0]:,.0f} DKK", f"{b['reached']} reached · {b['worried']} worried for nothing")
        st.markdown(f"**{int(y.sum())} of the {SIZE} students left.** You missed **{res['you'][1]['missed']}** of them.")
        st.dataframe(batch[batch["left"] == 1][["student_id", "programme", "submitted_share", "logins_last3", "risk"]]
                     .rename(columns={"risk": "model risk"}), hide_index=True, width="stretch")
        t = ss.totals
        st.caption(f"Total after {r + 1} round(s): you {t['you']:,.0f} · model {t['model']:,.0f} · Gus {t['gus']:,.0f} DKK "
                   "(500 DKK per conversation, 2,000 per false alarm, 30 % of leavers kept, 60,000 DKK each).")
        if st.button("➡️ Next batch of students", type="primary"):
            ss.round += 1; ss.done = False
            st.rerun()
