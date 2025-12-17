# State of Charge (SOC) estimation for electric Vertical Take-Off and Landing aircraft (eVTOL) batteries
--------------------
## 1. Introduction

### 1.1 State of Charge of batteries

리튬 이온 배터리는 전기차, 드론, 항공기 등 다양한 시스템에서 사용되는 중요한 에너지 저장 장치이다. 배터리를 안전하고 효율적으로 사용하기 위해서는 현재 배터리에 얼마나 많은 에너지가 남아있는지 정확하게 알아야 한다. 

이때 사용되는 대표적인 지표는 SOC (State of Charge)이다. SOC란 특정 온도와 discharge rate 등의 조건에서 배터리가 손상되지 않고 제공할 수 있는 최대 방전 용량을 의미한다. 이는 배터리의 nominal capacity (정격 용량)에 대해 남아있는 capacity의 비로 나타낸다. 여기서 discharge rate는 C-rate으로 표현되며, 배터리의 nominal capacity 대비 방전 전류의 크기를 의미한다. SOC를 나타내는 식은 아래와 같다.

$$ 
\mathrm{SOC} = \frac{Q_\text{remaining}}{Q_\text{nominal}}
$$


  SOC를 정확하게 아는 것은 운행 가능 시간 예측, 시스템 제어, 안전 관리 등을 위해 중요하다. 
그러나 실제 배터리 시스템에서 SOC를 직접적으로 측정할 수 없다. 배터리에서 실제로 측정 가능한 것은 전압(V), 전류(I), 온도(T)이며, 이를 통해 우리는 SOC를 예측해야 한다. 
남아있는 capacity의 양은 충전 혹은 방전 동안 흐르는 전류를 적분함으로써 구할 수 있다. 따라서 SOC는 다음과 같은 식으로도 표현된다.

$$
\mathrm{SOC}(t) = \mathrm{SOC}_0 - \frac{\eta \int_{t_0}^{t} i(t)\ dt}{Q_n}
$$

여기서 $\mathrm{SOC}_0$ 은 initial SOC를, $i(t)$는 전류를 의미한다. $\eta$는 efficiency factor이며 배터리의 충방전 과정에서 발생하는 에너지 손실을 반영한다. 실제로 배터리가 노화됨에 따라 $Q_n$ (i.e., 최대 가용 전하량)이 감소하고, charge/discharge rate, 온도, self-discharge, aging 등의 요소 역시 SOC에 영향을 미치기 때문에 이러한 부분을 고려한 보정이 필요하다. 

실험실 상황이 아닌 실제 배터리 운용 상황에서는 전류 적분에 기반한 SOC 계산에는 한계가 있다. 전류 측정 오차와 센서 노이즈로 인해 SOC의 오차가 시간에 따라 누적되는 문제가 발생할 수 있기 때문이다. 이러한 문제를 보완하기 위해 SOC 추정에 대한 다양한 연구가 진행되었다. 이는 배터리 내부 전기화학적 거동을 수학적으로 묘사하는 물리 기반 모델링과 머신러닝이나 딥러닝을 사용하는 data-driven 모델로 구분할 수 있다. 

----------------------
### 1.2 eVTOL

Electric Vertical Take-Off and Landing (eVTOL) aircraft는 전기 동력을 사용하는 항공기로, Urban air mobility의 발전에 따라 주목받고 있다. 기존의 fixed-wing aircraft와 달리 긴 활주로가 불필요하며 수직 이착륙과 호버링이 가능하기에 Automomous deliveries나 air taxies 등의 도심 교통, 응급 항공 서비스 및 기타 단거리 운송 분야에서 적합하다. 

eVTOL의 독특한 주행 특성은 급격하게 변동하는 power를 발생시키며, 이는 배터리 시스템에 큰 stress를 가한다. 또한 전기차(EV)와 비교했을 때 eVTOL은 더 엄격한 배터리 성능을 요구한다. 
특히 이착륙 과정에서 높은 discharge current가 필요하다. EV 배터리 실험에서는 주행 조건에 따라 C/16 - 2C 정도의 C-rate를 고려하는 반면, eVTOL의 경우 이착륙 시 1C - 20C 가량의 높은 C-rate가 요구된다. 착륙 과정에서는 배터리가 높은 power를 유지해야 하는데, 이미 배터리 셀은 소모된 상태에서 성능이 저하되므로 더욱 까다롭다. 또한 eVTOL은 복잡하고 가변적인 열, 기계적 환경에 노출되며 urban airspace 운용에서는 안전에 대한 요구사항이 더욱 엄격하다. 

따라서 eVTOL에서 배터리를 관리하는 것은 중요하다. 이는 비행 거리, 안전, mission reliability와 같은 핵심 요소를 직접적으로 결정하기 때문이다. 그러나 복잡한 운용 특성을 가지는 만큼 배터리 내부를 묘사하고 SOC를 예측하기가 어렵고, 기존의 전통적 모델링 방법은 이러한 eVTOL의 운행 상황에 적합하지 않을 수 있다. 따라서 현재는 eVTOL 배터리 내부 상태 추정을 위한 모델링 기법에 대한 연구가 활발히 이루어지고 있으며, 딥러닝과 물리 모델을 결합한 hybrid 방식의 모델링이 각광받고 있다. 

본 프로젝트에서는 eVTOL 배터리의 SOC를 예측하는 모델을 구현하는 것을 목표로 한다. 

----------------------------
## 2. Data description and Preprocessing

### 2.1 Data description
본 프로젝트에서는 Carnegi Mellon University에서 공개한 eVTOL 배터리 실험 데이터셋을 사용하였다. 
https://kilthub.cmu.edu/articles/dataset/eVTOL_Battery_Dataset/14226830

해당 데이터에는 eVTOL 항공기의 운용 환경을 모사한 전류 프로파일 하에서 리튬 이온 배터리를 반복적으로 충방전하며 측정된 실험 데이터이다. 
각 실험은 하나의 mission profile을 기준으로 수행되며, 하나의 사이클은 charging, rest, take-off, cruise, landing, rest의 단계로 구성된다. 특히 take-off와 landing 구간에서 요구되는 높은 전력을 반영하였다. 또한 50 cycle마다 배터리의 성능 평가를 위해 Reference Performance Test (RPT)를 수행하였다. 

Baseline 조건을 기준으로 총 22개의 셀에 대해 실험이 수행되었으며, 각 셀은 오직 하나의 실험 조건 (온도, 충전 프로토콜, 비행 프로파일 등)만이 변경되었다.  
예를 들어, VAH01.csv는 baseline에서 실험된 셀이며, VAH02.csv는 cruise time이 1000초로 증가된 조건 하에서 실험된 셀이다. 

---------------------------
### 2.2 Preprocessing
모델 학습을 위해 다음과 같은 전처리 과정을 수행하였다. 

1) 원본 데이터의 cycle number에 오류가 있어 새로운 cycle number를 부여하였다.
2) 측정에 오류가 있는 문제 사이클을 구분하기 위해 valid_cycle을 따로 표기하였다. 
3) mission 사이클만 학습시키기 위해 RPT 사이클을 따로 표기하였다.
4) 충방전 시 들어오고 나간 전하량을 이용하여 SOC를 계산하였다. 
   

아래 사진은 전처리 이후 데이터셋의 전압, 전류, 온도 그래프이다. (VAH01)

![VAH01 example](figures/VAH01_VIT.png)


아래는 모든 사이클에 대해 계산한 SOC의 그래프이다.

![VAH01 example](figures/soc_VAH01.png)

-----------------------------------------

## 3. Model and Training step

본 프로젝트에서는 SOC 추정을 위해 Random Forest regressor를 사용하였다. 
eVTOL 배터리의 경우 전압, 전류, 온도와 SOC의 관계가 명확한 선형 모델로 표현되기 어렵기 때문에 복잡한 비선형 관계를 학습할 수 있는 Random Forest가 적합하다고 판단하였다. 또한 RF는 별도의 하이퍼파라미터 튜닝 없이도 안정적인 성능을 보이고, 데이터 개수가 많지 않아도 과적합에 강하다는 장점이 있다. 

하나의 데이터 셋은 하나의 배터리 셀에 대해 초기 상태부터 열화가 진행될 때까지의 전체 과정을 포함한다. 따라서 데이터 포인트 단위가 아닌 셀 단위로 train/test split을 적용하였다. 

SOC추정은 RPT 사이클을 제외한 모든 valid한 사이클에 대해 추정되며, 여기서는 특히 discharge 구간의 SOC 추정에 초점을 맞춘다. Discharge 구간이 eVTOL의 비행 중 구간이며, 해당 구간에서의 SOC가 안전한 운용에 있어서 더 중요하다고 판단하였기 때문이다. 

SOC 추정을 위해 사용한 입력 변수는 전압, 전류, 온도이다. 전압은 SOC와 가장 직접적인 관계를 가지는 물리량이며, 전류는 방전 상태와 부하 조건을 반영한다. 또한 배터리 내부 저항 및 전기화학 반응은 온도에 크게 의존하므로 온도를 입력변수로 포함하였다. 
추가적인 신호 처리 없이 실제로 측정 가능한 기본적인 물리량만을 입력으로 사용하였다. 이는 실제 운용 환경에서의 적용 가능성을 고려한 선택이다. 

### 3.1 basic model
입력 변수로 전압, 전류, 온도만을 사용한 모델이다. 

```python
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# VAH## 파일이 들어있는 폴더 
DATA_DIR = r"D:\eVTOL dataset\ML project" 

# 사용할 파일 목록 20개
vah_list = [
    "VAH01","VAH02","VAH06","VAH07","VAH09","VAH10","VAH11","VAH12","VAH13",
    "VAH15","VAH16","VAH17","VAH20","VAH22","VAH24","VAH25","VAH26","VAH27", "VAH28", "VAH30"] 


# column 이름
t_col     = "time_s"
v_col     = "Ecell_V"
i_col     = "I_mA"
temp_col  = "Temperature__C"
cycle_col = "new_cycle"

valid_col = "valid_cycle"
rpt_col   = "RPT_cycle"

soc_col   = "SOC_shifted"

-----------------------------------------------------------------------
# 전체 파일 일기
  # 파일을 모두 읽어서 각 row가 어느 파일에서 왔는지 표기한다.
  # 이후 셀 단위로 train/test set을 구분할 것이다.

def load_cells(data_dir, cell_names):
    dfs = []
    for cell in cell_names:
        fp = os.path.join(data_dir, f"{cell}.csv")
        d = pd.read_csv(fp)
        d["cell_id"] = cell
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)

df = load_cells(DATA_DIR, vah_list)

----------------------------------------------------------------------
# SOC 추정 사이클만 mask
  # SOC 추정을 할 사이클은 valid 한 사이클 중 RPT 사이클이 아닌 사이클이며, 
  # discharge 구간에서만 SOC를 추정한다.

mask = (
    (df[valid_col] == True) &
    (df[rpt_col] == False) &
    (df[i_col] < 0)
)

use_cols = ["cell_id", cycle_col, t_col, v_col, i_col, temp_col, soc_col]
dfd = df.loc[mask, use_cols].copy()
dfd = dfd.dropna(subset=use_cols)
dfd = dfd.sort_values(["cell_id", cycle_col, t_col])

--------------------------------------------------------------------
# SOC cliping
  # 물리적으로 불가능한 SOC 이상치가 있을 경우를 대비하여 clip한다.

dfd[soc_col] = dfd[soc_col].astype(float).clip(0.0, 1.0)

---------------------------------------------------------------------
# 사이클 번호 재설정
  # valid, non-RPT 사이클만 남긴 후 cycle number가 연속적으로 증가할 수 있도록 재설정한다.

pairs = (
    dfd[["cell_id", cycle_col]]
    .drop_duplicates()
    .sort_values(["cell_id", cycle_col])
)
pairs["cycle_ml"] = pairs.groupby("cell_id").cumcount() + 1

dfd = dfd.merge(pairs, on=["cell_id", cycle_col], how="left")

------------------------------------------------------------------
# training/test spliting
  # 전체 20개의 파일 중, 20개를 랜덤하게 골라서 training에 사용한다.

all_cells = sorted(dfd["cell_id"].unique())
rng = np.random.RandomState(42)

n_test = max(1, int(round(0.2 * len(all_cells))))  # 20개면 보통 4개
test_cells = sorted(rng.choice(all_cells, size=n_test, replace=False))
train_cells = [c for c in all_cells if c not in test_cells]

train_df = dfd.loc[dfd["cell_id"].isin(train_cells)].copy()
test_df  = dfd.loc[dfd["cell_id"].isin(test_cells)].copy()

print("Train cells:", train_cells)
print("Test cells :", test_cells)
print("Train rows :", len(train_df), " Test rows:", len(test_df))


----------------------------------------------------------------
# 모델 학습
  # 입력은 전압, 전류, 온도를 사용한다.

X_train = train_df[[v_col, i_col, temp_col]].to_numpy(dtype=float)
y_train = train_df[soc_col].to_numpy(dtype=float)

X_test  = test_df[[v_col, i_col, temp_col]].to_numpy(dtype=float)
y_test  = test_df[soc_col].to_numpy(dtype=float)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=3
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Test MAE  = {mae:.5f}")
print(f"Test RMSE = {rmse:.5f}")

------------------------------------------------------------------
# 각 파일 별 MAE

test_df = test_df.copy()
test_df["pred"] = y_pred
cell_mae = (
    test_df.groupby("cell_id")
           .apply(lambda g: mean_absolute_error(g[soc_col].to_numpy(), g["pred"].to_numpy()))
           .sort_values()
)
print("\nMAE by test cell:")
print(cell_mae)
```
----------------------------

### 3.2 Extended model

입력 변수로 전압, 전류, 온도, 전압 변화율 (dV/dt), 시간 간격 (dt)를 고려한 모델이다.
실제 배터리에서 측정되는 단자 전압에는 각 SOC에 따른 open circuit voltage (이상적인 단자 전압), ohmic resistance로 인한 내부 전압 강하, 과거 전류 히스토리에 따른 polarization이 모두 영향을 준다. 이를 고려하기 위하여 dV/dt를 입력변수로 설정하였다. dV/dt는 전압이 현재 어떻게 변하는지를 알려주는데, 예를 들어 dV/dt가 크다면 전압이 빠르게 떨어지는 있음을 뜻한다. 또한 본 프로젝트에서 사용한 데이터 분석 결과, 데이터의 측정 간격이 일정하지 않았음을 알 수 있었다. 따라서 입력 변수로 dt를 함께 줌으로써 전압의 변화율이 몇 초 동안 관측된 것인지 모델이 판단할 수 있도록 하였다. 

아래의 코드는 3.1에서의 

```python

# 확인할 사이클의 사이클 넘버를 연속적으로.
pairs = (
    dfd[["cell_id", cycle_col]]
    .drop_duplicates()
    .sort_values(["cell_id", cycle_col])
)
pairs["cycle_ml"] = pairs.groupby("cell_id").cumcount() + 1

dfd = dfd.merge(pairs, on=["cell_id", cycle_col], how="left")
dfd = dfd.sort_values(["cell_id", "cycle_ml", t_col]).reset_index(drop=True)


# 시간 중복된 측정이 있으면 dt=0이 생기기 때문에 dV/dt가 0이 될 수 있음. 
# 따라서 중복된 측정이 있다면 제거. 
dfd = dfd.drop_duplicates(subset=["cell_id", "cycle_ml", t_col], keep="first")


# dV/dt를 고려하자. 
grp = dfd.groupby(["cell_id", "cycle_ml"], sort=False)

dfd["dt"] = grp[t_col].diff()
dfd["dV"] = grp[v_col].diff()

# dt==0 방지
dfd["dt"] = dfd["dt"].replace(0, np.nan)

dfd["dVdt"] = dfd["dV"] / dfd["dt"]

# inf/NaN 처리: 각 cycle 첫 행은 NaN이므로 0으로
dfd["dVdt"] = dfd["dVdt"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
dfd["dt"]   = dfd["dt"].fillna(0.0)

# outlier가 있을 수 있으니까 방지하자
dVdt_clip = 0.05  # V/s
dt_clip   = 60.0  # s
dfd["dVdt"] = dfd["dVdt"].clip(-dVdt_clip, dVdt_clip)
dfd["dt"]   = dfd["dt"].clip(0.0, dt_clip)


# 이전과 동일하게 train/test 셋 설정
all_cells = sorted(dfd["cell_id"].unique())
rng = np.random.RandomState(42)

n_test = max(1, int(round(0.2 * len(all_cells))))  # 20개면 보통 4개
test_cells = sorted(rng.choice(all_cells, size=n_test, replace=False))
train_cells = [c for c in all_cells if c not in test_cells]

train_df = dfd.loc[dfd["cell_id"].isin(train_cells)].copy()
test_df  = dfd.loc[dfd["cell_id"].isin(test_cells)].copy()

print("Train cells:", train_cells)
print("Test cells :", test_cells)
print("Train rows :", len(train_df), " Test rows:", len(test_df))


# 모델 학습:
# 입력 시 추가로 dV/dt와 dt를 고려한다.
feat_cols = [v_col, i_col, temp_col, "dVdt", "dt"]

X_train = train_df[feat_cols].to_numpy(dtype=float)
y_train = train_df[soc_col].to_numpy(dtype=float)

X_test  = test_df[feat_cols].to_numpy(dtype=float)
y_test  = test_df[soc_col].to_numpy(dtype=float)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=3
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Test MAE  = {mae:.5f}")
print(f"Test RMSE = {rmse:.5f}")

# 셀 별 MAE
test_df = test_df.copy()
test_df["pred"] = y_pred

cell_mae = (
    test_df.groupby("cell_id")
           .apply(lambda g: mean_absolute_error(g[soc_col].to_numpy(), g["pred"].to_numpy()))
           .sort_values()
)

print("\nMAE by test cell:")
print(cell_mae)


# 4. Results and Discussion

## 5. reference
