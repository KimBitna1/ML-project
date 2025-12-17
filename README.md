# State of Charge (SOC) estimation for electric Vertical Take-Off and Landing aircraft (eVTOL) batteries

## 1. Introduction
### 1.1 State of Charge of batteries
리튬 이온 배터리는 전기차, 드론, 항공기 등 다양한 시스템에서 사용되는 중요한 에너지 저장 장치이다. 배터리를 안전하고 효율적으로 사용하기 위해서는 현재 배터리에 얼마나 많은 에너지가 남아있는지 정확하게 알아야 한다. 

이때 사용되는 대표적인 지표는 SOC (State of Charge)이다. SOC란 특정 온도와 discharge rate (한 시간 동안 사용할 수 있는 전류량)등의 조건에서 배터리가 손상되지 않고 제공할 수 있는 최대 방전 용량을 의미한다. 이는 배터리의 nominal capacity (최대 가용 전하량)에 대해 남아있는 capacity의 비로 나타낸다.

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
Electric Vertical Take-Off and Landing (eVTOL) aircraft는 전기 동력을 사용하는 항공기로, Urban air mobility의 발전에 따라 주목받고 있다. Automomous deliveries 혹은 air taxies 등의 단거리 이동 수요를 해결할 가능성이 있다. eVTOL은 전기차 (EV)와 달리 비행 중 수직 이륙 및 착륙, 호버링과 같은 독특한 주행 특성을 가진다. 이로 인해 급격한 변동의 power가 요구되며 이는 배터리 시스템에 큰 stress를 가한다. 
