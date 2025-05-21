
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Название приложения
st.title("Приложение для поддержки принятия решения по выбору рационального варианта верхнего строения морской нефтегазопромысловой платформы")

# Ввод данных
st.header("Ввод данных")
num_alternatives = st.number_input("Количество альтернатив", min_value=2, value=3)
criteria = ["ЧДД (млн руб.)", "ВНД (%)", "Масса ВСП (т)", "Срок строительства ВСП (мес.)", "Пиковая мощность (МВт)", "Выбросы CO2 (т)"]
benefit_criteria = [True, True, False, False, False, False]

weights_linguistic = {
    "Очень низкая": (0.0, 0.0, 0.1),
    "Низкая": (0.0, 0.1, 0.3),
    "Средняя": (0.2, 0.5, 0.8),
    "Высокая": (0.7, 0.9, 1.0),
    "Очень высокая": (0.9, 1.0, 1.0)
}

st.subheader("Весовые коэффициенты критериев (лингвистические)")
weights = []
for crit in criteria:
    sel = st.selectbox(f"Вес для критерия '{crit}'", list(weights_linguistic.keys()), index=2)
    weights.append(weights_linguistic[sel])
weights = np.array(weights)

st.subheader("Оценки альтернатив по критериям (треугольные числа)")
data = []
for i in range(num_alternatives):
    row = []
    st.markdown(f"**Альтернатива {i+1}**")
    for crit in criteria:
        val = st.text_input(f"{crit} (формат: l,m,u)", key=f"{i}_{crit}")
        try:
            tri = tuple(map(float, val.split(",")))
        except:
            tri = (0.0, 0.0, 0.0)
        row.append(tri)
    data.append(row)
data = np.array(data)

# Нормализация
def normalize(data, benefit):
    norm = []
    for j in range(data.shape[1]):
        col = data[:, j]
        if benefit[j]:
            max_val = np.max(col[:, 2])
            norm_col = [(l/max_val, m/max_val, u/max_val) for l, m, u in col]
        else:
            min_val = np.min(col[:, 0])
            norm_col = [(min_val/u, min_val/m, min_val/l) if l != 0 else (0, 0, 0) for l, m, u in col]
        norm.append(norm_col)
    return np.array(norm).T

norm_data = normalize(data, benefit_criteria)

# Взвешенная нормализация
def weighted_fuzzy_decision(norm_data, weights):
    weighted = []
    for i in range(norm_data.shape[0]):
        weighted_row = []
        for j in range(norm_data.shape[1]):
            w = weights[j]
            r = norm_data[i, j]
            weighted_row.append((r[0]*w[0], r[1]*w[1], r[2]*w[2]))
        weighted.append(weighted_row)
    return np.array(weighted)

weighted_data = weighted_fuzzy_decision(norm_data, weights)

# Идеальные решения
def ideal_solutions(weighted_data):
    n_criteria = weighted_data.shape[1]
    pis = [(np.max(weighted_data[:, j, 2]),
            np.max(weighted_data[:, j, 1]),
            np.max(weighted_data[:, j, 0])) for j in range(n_criteria)]
    nis = [(np.min(weighted_data[:, j, 0]),
            np.min(weighted_data[:, j, 1]),
            np.min(weighted_data[:, j, 2])) for j in range(n_criteria)]
    return pis, nis

pis, nis = ideal_solutions(weighted_data)

# Расстояния
def distance(a, b):
    return np.sqrt((1/3)*((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2))

def closeness(weighted_data, pis, nis):
    cc = []
    for i in range(weighted_data.shape[0]):
        d_pos = np.sqrt(np.sum([distance(weighted_data[i, j], pis[j])**2 for j in range(weighted_data.shape[1])]))
        d_neg = np.sqrt(np.sum([distance(weighted_data[i, j], nis[j])**2 for j in range(weighted_data.shape[1])]))
        cc.append(d_neg / (d_pos + d_neg))
    return cc

cc_scores = closeness(weighted_data, pis, nis)

# Итоги
st.header("Результаты")
results_df = pd.DataFrame({
    "Альтернатива": [f"Альтернатива {i+1}" for i in range(num_alternatives)],
    "Коэффициент близости": cc_scores
})
results_df["Ранг"] = results_df["Коэффициент близости"].rank(ascending=False).astype(int)
results_df = results_df.sort_values("Ранг")
st.dataframe(results_df)

# Визуализация: Столбчатая диаграмма
st.subheader("Визуализация результатов")
fig_bar = px.bar(results_df, x="Альтернатива", y="Коэффициент близости", color="Альтернатива", title="Ранжирование альтернатив")
st.plotly_chart(fig_bar)

# TFN-график
st.subheader("TFN по первому критерию")
fig_tfn, ax = plt.subplots()
for i in range(num_alternatives):
    tri = data[i, 0]
    ax.plot([tri[0], tri[1], tri[2]], [0, 1, 0], label=f"Альтернатива {i+1}")
ax.set_title("TFN по критерию 'ЧДД (млн руб.)'")
ax.legend()
st.pyplot(fig_tfn)

# Радар-график
st.subheader("Радар-график по средним значениям критериев")
mean_data = np.mean([[m for l, m, u in row] for row in data], axis=0)
alt_labels = [f"Альтернатива {i+1}" for i in range(num_alternatives)]
fig_radar = px.line_polar(r=pd.DataFrame(data[:, :, 1], columns=criteria, index=alt_labels),
                          line_close=True)
st.plotly_chart(fig_radar)
