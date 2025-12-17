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
4) 충방전 시 들어오고 나간 전하량을 이용하여 각 사이클 별 SOC를 계산하였다.

아래 사진은 전처리 이후 데이터셋의 모든 사이클에 대한 그래프이다.
![VAH01 example](VAH01_all.png)


## 3. Model and Training step

## 4. Results and Discussion
