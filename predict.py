import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True, help=' : name of the lecture') 
parser.add_argument('--lim', help=' : custom capacity of the lecture', default=-1)
parser.add_argument('--cnt', help=' : number of years to draw into the figure', default=-1)
args = parser.parse_args()

plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['xtick.minor.width'] = 0

plt.rcParams['font.family'] ='HCR Dotum'
plt.rcParams['axes.unicode_minus'] = False

model = tf.keras.models.load_model("Models/main.keras", safe_mode=False)

def process(file):
    df = pd.read_excel(file)
    df.columns = df.iloc[1]
    df = df[2:].reset_index(drop=True)
    
    df["전체정원"] = df["정원"].apply(lambda x: int(x.split("(")[0]))
    df["장바구니신청"] = df["장바구니신청"].astype(int)
    df["경쟁률"] = df["장바구니신청"] / df["전체정원"]
    df = df[df["전체정원"]!=0]
    df.drop(columns = ["교과목번호", "강좌번호", "부제명", "수업형태", "강의실(동-호)(#연건, *평창)", "재학생장바구니신청",
                      "신입생장바구니신청", "비고", "강의언어", "개설상태", "정원", "수강신청인원"], inplace=True)
    df["연도"] = int(file.split("/")[2][:4])
    df["학기"] = int(file.split("_")[1][0])
    df["학년"] = df["학년"].apply(lambda x: int(str(x)[0] if str.isdigit(str(x)[0]) else 0))
    return df

def semester_to_int(row):
    return row['연도'] * 10 + (1 if row['학기'] == 1 else 2)

def getDataFrame():
    files = [join("./Data/", f) for f in listdir("./Data") if isfile(join("./Data", f))]
    df = []
    for file in files:
        df.append(process(file))
    
    df = pd.concat(df)
    df = df.reset_index(drop=True)
    
    for col in ["학점", "강의", "실습"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[["학점", "강의", "실습"]] = df[["학점", "강의", "실습"]].fillna(0)

    df["수업교시"] = df["수업교시"].fillna(0)
    df["수업교시"] = df["수업교시"].apply(lambda x: int(x.split("(")[1].split(":")[0]) + int(x.split(":")[1].split("~")[0]) / 60 if type(x)==str else x)
    
    df['semester_idx'] = df.apply(semester_to_int, axis=1)
    
    df['row_index'] = df.index
    df_sorted = df.sort_values(by=["교과목명", "semester_idx"]).reset_index(drop=True)

    return df, df_sorted

df, df_sorted = getDataFrame()

categorical_features = ["교과구분", "이수과정"]

numerical_features = ["학년", "학점", "강의", "실습", "수업교시", "장바구니신청", "전체정원"]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

df_clean = df[categorical_features + numerical_features].copy()
X_all = preprocessor.fit_transform(df_clean)
temp = X_all[:, -2].copy()
X_all[:, -2] = X_all[:, -1]
X_all[:, -1] = temp
X_all_sorted = X_all[df_sorted["row_index"].values]

window_size = 3

def get_result(custom_lecture, custom_lim=-1, graph_cnt=-1):
    global dense_features, LSTM_features, FIN_features, window_size

    if not (custom_lecture in df_sorted["교과목명"].unique()):
        raise ValueError('Cannot find lecture.')
    
    datas = df_sorted[df_sorted["교과목명"]==custom_lecture]
    semesters = datas["semester_idx"].unique()
    
    sem_data_list = []
    
    for i in range(len(semesters)):
        sem_data = datas[datas["semester_idx"]==semesters[i]]
        sem_data = X_all_sorted[sem_data.index.to_list()]
        sem_data_list.append(sem_data.mean(axis=0))
    
    sem_data_list = np.array(sem_data_list)

    if custom_lim == -1:
        custom_lim = sem_data_list[-1, -2]

    temp_lim = np.full((sem_data_list.shape[0], 1), custom_lim)
    sem_data_list = np.concatenate([sem_data_list, temp_lim], axis=1)
    
    if len(semesters) < window_size + 1:
        y_pred = np.array([[sem_data_list[-1, -2]]])
    else:
        custom_data = np.expand_dims(sem_data_list[-window_size:], axis=0)
        y_pred = model.predict(custom_data, verbose=0)

    semester_sum = np.vectorize(lambda x: str(x)[:-1] + "-" + str(x)[-1])(semesters)
    semester_sum = np.concatenate([semester_sum, ["다음 학기"]])
    
    x_index = range(len(semester_sum))
    bar_width = 0.35

    lim = np.concatenate([sem_data_list[:, -3], [custom_lim]])
    res = np.concatenate([sem_data_list[:, -2], y_pred[0]])
    ratio = res / lim

    if graph_cnt != -1:
        x_index = range(graph_cnt)
        semester_sum = semester_sum[-graph_cnt:]
        lim = lim[-graph_cnt:]
        res = res[-graph_cnt:]
        ratio = ratio[-graph_cnt:]

    fig, ax1 = plt.subplots(figsize=(6, 3))
    bars1 = ax1.bar(semester_sum, lim, width=bar_width, label='전체정원')
    bars2 = ax1.bar([i + bar_width for i in x_index], res, width=bar_width, label='장바구니신청')
    ax1.set_xlabel('학기')
    ax1.set_ylabel('학생 수')
    ax1.set_xticks([i + bar_width / 2 for i in x_index], semester_sum)

    ax2 = ax1.twinx()
    line, = ax2.plot([i + bar_width / 2 for i in x_index], ratio, label='경쟁률', color='green', marker='o', linewidth=2, zorder=1)
    ax2.set_ylabel('경쟁률')
    legend = ax1.legend(loc='upper left')
    legend.set_zorder(100)
    legend = ax2.legend(loc = 'lower right')
    legend.set_zorder(10)

    plt.title(f'{custom_lecture}의 경쟁률 예측')
    
    plt.tight_layout()
    plt.savefig("Figures/predict_result.png")
    
    return lim[-1], res[-1], ratio[-1]

lecture_name = args.name
lim, res, ratio = get_result(lecture_name, int(args.lim), int(args.cnt))
print(f"[{lecture_name}]의 예측\n정원: {int(lim)}명\n예측 장바구니인원: {res:.2f}명\n예측 경쟁률: {ratio:.2f}")
print("자세한 결과는 Figures/predict_result.png에서 확인 가능합니다.")